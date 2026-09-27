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
    ):
        self.sandbox = sandbox or VerificationSandbox(default_timeout=0.75, reduced_timeout=0.30, workers=workers)
        self.parser = parser or WorkedExampleParser()
        self.alpha = default_alpha

    def close(self) -> None:
        self.sandbox.close()

    # ------------------------------------------------------------------ #
    def verify_query(self, query_text: str, candidate_ids: List[str], corpus_dict: Dict[str, str]) -> QueryVerification:
        """Run every candidate on the query's examples; return alpha-independent confidences."""
        examples = self.parser.parse_examples(query_text)
        if not examples:
            return QueryVerification(n_examples=0)
        any_order = statement_allows_any_order(query_text)
        results = self.sandbox.verify_many(
            [(doc_id, corpus_dict.get(doc_id, "")) for doc_id in candidate_ids], examples, multiline_set=any_order
        )
        return self.summarize(results, len(examples))

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
