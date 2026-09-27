"""
VERA Rarity-Weighted Verification Boost & Top-K Re-Ranker (Rung R2)
==================================================================
1. Rarity-weighted confidence
       conf(d, q) = (e_pass / E) * 1 / (1 + log2(m))
   where m is the number of top-K candidates whose output on example 1 matches the
   expected output (a bare "4" that half the pool prints is worth little; a 47-line
   exact match is near-proof).
2. Score blending
       S_final(d, q) = norm(S_dense(d, q)) + alpha * conf(d, q)
   with dense scores min-max normalised over the candidate pool. The boost is additive
   and bounded: non-passers are never filtered, a decisive dense margin is never
   overridden by a low-confidence pass.
3. alpha is grid-fit on the 500-pair dev split (``scripts/m7_topk_verify.py``).

Verification of the K candidates runs in parallel on the sandbox worker pool. The
expensive part (running code) is separated from the cheap part (blending with alpha)
so the dev grid search verifies each query exactly once.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from vera.verify.executor import CandidateVerificationResult, VerificationSandbox
from vera.verify.parser import ExamplePair, WorkedExampleParser, statement_allows_any_order


def compute_rarity_confidence(passed_examples: int, total_examples: int, identical_output_count: int) -> float:
    """conf = (e_pass / E) * 1 / (1 + log2(m)), clipped to [0, 1]."""
    if total_examples <= 0 or passed_examples <= 0:
        return 0.0
    pass_ratio = passed_examples / total_examples
    m = max(1, identical_output_count)
    conf = pass_ratio / (1.0 + math.log2(m))
    return min(1.0, max(0.0, conf))


def min_max_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Normalize score values into [0, 1] across the candidate pool."""
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


@dataclass
class QueryVerification:
    """Everything the boost needs for one query, computed once (alpha-independent)."""

    n_examples: int
    conf: Dict[str, float] = field(default_factory=dict)        # doc_id -> rarity-weighted confidence
    passed: Dict[str, int] = field(default_factory=dict)        # doc_id -> examples passed
    m_first: int = 0                                            # candidates matching example 1
    n_all_pass: int = 0                                         # candidates passing every example
    runtime_ms: float = 0.0
    statuses: Dict[str, int] = field(default_factory=dict)      # status histogram over first runs


@dataclass
class CandidateBoostInfo:
    doc_id: str
    dense_score: float
    norm_dense_score: float
    confidence: float
    final_score: float
    passed_examples: int
    total_examples: int
    identical_output_m: int
    first_output: str


class TopKVerifier:
    """Verifies top-K dense candidates and applies the calibrated rarity boost."""

    def __init__(
        self,
        sandbox: Optional[VerificationSandbox] = None,
        parser: Optional[WorkedExampleParser] = None,
        default_alpha: float = 0.20,
        workers: Optional[int] = None,
        cache_path: Optional[str] = None,
    ):
        self.sandbox = sandbox or VerificationSandbox(default_timeout=0.75, reduced_timeout=0.30, workers=workers)
        self.parser = parser or WorkedExampleParser()
        self.alpha = default_alpha
        # Optional on-disk cache of alpha-independent verification results, keyed by (query text, candidate ids).
        # Lets the test split be verified once and re-blended with any alpha (e.g. after the dev fit).
        self.cache_path = cache_path
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_dirty = 0
        if cache_path:
            import json
            import os

            if os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)

    def close(self) -> None:
        self.flush_cache()
        self.sandbox.close()

    def flush_cache(self) -> None:
        if self.cache_path and self._cache_dirty:
            import json

            tmp = self.cache_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._cache, f)
            import os

            os.replace(tmp, self.cache_path)
            self._cache_dirty = 0

    @staticmethod
    def _cache_key(query_text: str, candidate_ids: List[str]) -> str:
        import hashlib

        h = hashlib.sha1(query_text.encode("utf-8", errors="ignore"))
        h.update(("|" + ",".join(candidate_ids)).encode("utf-8"))
        return h.hexdigest()

    # ------------------------------------------------------------------ #
    def verify_query(self, query_text: str, candidate_ids: List[str], corpus_dict: Dict[str, str]) -> QueryVerification:
        """Run every candidate on the query's examples; return alpha-independent confidences."""
        key = self._cache_key(query_text, candidate_ids) if self.cache_path else None
        if key is not None and key in self._cache:
            d = self._cache[key]
            qv = QueryVerification(n_examples=d["n_examples"])
            qv.conf, qv.passed, qv.m_first, qv.n_all_pass = d["conf"], d["passed"], d["m_first"], d["n_all_pass"]
            qv.runtime_ms, qv.statuses = d.get("runtime_ms", 0.0), d.get("statuses", {})
            return qv
        examples = self.parser.parse_examples(query_text)
        if not examples:
            qv = QueryVerification(n_examples=0)
        else:
            any_order = statement_allows_any_order(query_text)
            results = self.sandbox.verify_many(
                [(doc_id, corpus_dict.get(doc_id, "")) for doc_id in candidate_ids], examples, multiline_set=any_order
            )
            qv = self.summarize(results, len(examples))
        if key is not None:
            self._cache[key] = {"n_examples": qv.n_examples, "conf": qv.conf, "passed": qv.passed, "m_first": qv.m_first,
                                "n_all_pass": qv.n_all_pass, "runtime_ms": qv.runtime_ms, "statuses": qv.statuses}
            self._cache_dirty += 1
            if self._cache_dirty >= 200:
                self.flush_cache()
        return qv

    @staticmethod
    def summarize(results: Dict[str, CandidateVerificationResult], n_examples: int) -> QueryVerification:
        qv = QueryVerification(n_examples=n_examples)
        # m = number of candidates whose output on example 1 matched (i.e. passed example 1)
        first_pass = [doc for doc, r in results.items() if r.results and r.results[0].matched]
        qv.m_first = len(first_pass)
        statuses: Counter = Counter()
        for doc_id, r in results.items():
            qv.passed[doc_id] = r.passed_examples
            qv.conf[doc_id] = compute_rarity_confidence(r.passed_examples, n_examples, qv.m_first if r.passed_examples > 0 else 1)
            qv.runtime_ms += r.total_runtime_ms
            if r.all_passed:
                qv.n_all_pass += 1
            if r.results:
                statuses[r.results[0].status] += 1
        qv.statuses = dict(statuses)
        return qv

    @staticmethod
    def blend(dense_scores: Dict[str, float], qv: QueryVerification, alpha: float, top_k: int) -> Dict[str, float]:
        """S_final = norm(dense) + alpha * conf for the verified top-K; tail keeps norm(dense)."""
        norm = min_max_normalize(dense_scores)
        ordered = sorted(dense_scores.items(), key=lambda kv: kv[1], reverse=True)
        out: Dict[str, float] = {}
        for i, (doc_id, _) in enumerate(ordered):
            conf = qv.conf.get(doc_id, 0.0) if i < top_k else 0.0
            out[doc_id] = norm[doc_id] + alpha * conf
        return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))

    # ------------------------------------------------------------------ #
    def rerank_query(
        self,
        query_text: str,
        dense_scores: Dict[str, float],
        corpus_dict: Dict[str, str],
        top_k: int = 150,
        alpha: Optional[float] = None,
        return_verification: bool = False,
    ):
        """Re-rank one query: verify its dense top-K, then blend with alpha."""
        effective_alpha = self.alpha if alpha is None else alpha
        ordered = sorted(dense_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_ids = [doc_id for doc_id, _ in ordered[:top_k]]
        qv = self.verify_query(query_text, top_ids, corpus_dict)
        blended = self.blend(dense_scores, qv, effective_alpha, top_k)
        if return_verification:
            return blended, qv
        return blended


class GatedVerifier(TopKVerifier):
    """R3: always verify the dense top ``base_k``; for *uncertain* queries (dense top-1/top-2 margin below
    ``tau``) extend verification to dense ranks ``base_k..extend_k`` filtered by the static signature gate.

    ``gate`` is a :class:`vera.gate.router.SignatureGate` (or None to skip the layout filter).
    """

    def __init__(self, base_k: int = 150, extend_k: int = 1000, tau: float = 0.02, gate: Any = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.base_k = base_k
        self.extend_k = extend_k
        self.tau = tau
        self.gate = gate
        self.stats = {"queries": 0, "extended": 0, "extra_candidates": 0}

    def candidate_ids(self, query_text: str, ordered: List[Tuple[str, float]]) -> List[str]:
        ids = [d for d, _ in ordered[: self.base_k]]
        margin = ordered[0][1] - ordered[1][1] if len(ordered) > 1 else float("inf")
        self.stats["queries"] += 1
        if margin < self.tau and len(ordered) > self.base_k:
            tail = [d for d, _ in ordered[self.base_k: self.extend_k]]
            if self.gate is not None:
                examples = self.parser.parse_examples(query_text)
                if examples:
                    allowed = set(self.gate.candidates(examples[0].stdin))
                    tail = [d for d in tail if d in allowed]
            ids.extend(tail)
            self.stats["extended"] += 1
            self.stats["extra_candidates"] += len(tail)
        return ids

    def rerank_query(self, query_text, dense_scores, corpus_dict, top_k=150, alpha=None, return_verification=False):
        effective_alpha = self.alpha if alpha is None else alpha
        ordered = sorted(dense_scores.items(), key=lambda kv: kv[1], reverse=True)
        ids = self.candidate_ids(query_text, ordered)
        qv = self.verify_query(query_text, ids, corpus_dict)
        blended = self.blend(dense_scores, qv, effective_alpha, top_k=len(ordered))  # conf is 0 for unverified docs
        if return_verification:
            return blended, qv
        return blended
