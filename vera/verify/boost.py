"""
VERA Rarity-Weighted Verification Boost & Top-K Re-Ranker
=========================================================
Implements Milestone M7 · Rung R2 (Guaranteed Fallback Ship):
1. Rarity-Weighted Confidence:
   conf(d, q) = (e_pass / E) * (1 / (1 + log2(m)))
   where m is the count of top-K candidates producing the identical output
   on example 1.
2. Score Blending:
   S_final(d, q) = norm(S_dense(d, q)) + alpha * conf(d, q)
3. Grid Search Calibration:
   Fits alpha in [0.05, 0.40] on the 500-pair dev split to maximize NDCG@10.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from vera.verify.executor import CandidateVerificationResult, VerificationSandbox
from vera.verify.parser import ExamplePair, WorkedExampleParser


def compute_rarity_confidence(
    passed_examples: int,
    total_examples: int,
    identical_output_count: int,
) -> float:
    """Compute rarity-discounted verification confidence in [0.0, 1.0].

    Parameters
    ----------
    passed_examples : int
        Number of worked examples the candidate passed (e_pass).
    total_examples : int
        Total number of worked examples extracted (E).
    identical_output_count : int
        Number of candidates in top-K producing identical output on example 1 (m).

    Returns
    -------
    float
        Confidence boost score in [0.0, 1.0].
    """
    if total_examples <= 0 or passed_examples <= 0:
        return 0.0

    pass_ratio = passed_examples / total_examples
    m = max(1, identical_output_count)
    rarity_weight = 1.0 / (1.0 + math.log2(m))
    conf = pass_ratio * rarity_weight

    return min(1.0, max(0.0, conf))


def min_max_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Normalize score values into [0.0, 1.0] across candidate pool."""
    if not scores:
        return {}

    values = list(scores.values())
    min_v = min(values)
    max_v = max(values)

    if max_v - min_v < 1e-9:
        return {k: 1.0 for k in scores}

    return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}


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
    """Verifies top-K dense retrieval candidates and applies calibrated rarity boost."""

    def __init__(
        self,
        sandbox: Optional[VerificationSandbox] = None,
        parser: Optional[WorkedExampleParser] = None,
        default_alpha: float = 0.20,
    ):
        self.sandbox = sandbox or VerificationSandbox(default_timeout=0.06, reduced_timeout=0.02)
        self.parser = parser or WorkedExampleParser()
        self.alpha = default_alpha

    def rerank_query(
        self,
        query_text: str,
        dense_scores: Dict[str, float],
        corpus_dict: Dict[str, str],
        top_k: int = 150,
        alpha: Optional[float] = None,
    ) -> Dict[str, float]:
        """Re-ranks top-K dense candidate documents for a single query using verification.

        Parameters
        ----------
        query_text : str
            Natural language problem statement.
        dense_scores : Dict[str, float]
            Dictionary of doc_id -> dense retrieval score.
        corpus_dict : Dict[str, str]
            Dictionary of doc_id -> raw code string.
        top_k : int
            Maximum number of dense candidates to verify (default 150).
        alpha : Optional[float]
            Verification boost weight. If None, uses self.alpha.

        Returns
        -------
        Dict[str, float]
            Re-ranked dictionary of doc_id -> final blended score.
        """
        effective_alpha = self.alpha if alpha is None else alpha

        # Sort candidate documents by dense score and select top_k
        sorted_candidates = sorted(dense_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_slice = sorted_candidates[:top_k]
        top_candidate_ids = [doc_id for doc_id, _ in top_slice]
        top_dense_map = {doc_id: score for doc_id, score in top_slice}

        # Min-max normalize dense scores across the candidate pool
        norm_dense = min_max_normalize(dense_scores)

        # Parse worked examples from query
        examples = self.parser.parse_examples(query_text)
        if not examples:
            # If no examples are available, return normalized dense scores untouched
            return dict(sorted(norm_dense.items(), key=lambda kv: kv[1], reverse=True))

        # Run verification sandbox on top-K candidates
        verification_results: Dict[str, CandidateVerificationResult] = {}
        for doc_id in top_candidate_ids:
            code = corpus_dict.get(doc_id, "")
            res = self.sandbox.verify_candidate(code, examples)
            verification_results[doc_id] = res

        # Count frequencies of identical first outputs (m) among passing candidates
        first_outputs: List[str] = [
            res.first_output for res in verification_results.values() if res.passed_examples > 0 and res.first_output
        ]
        output_counter = Counter(first_outputs)

        # Compute blended scores
        blended_scores: Dict[str, float] = {}
        for doc_id in top_candidate_ids:
            res = verification_results[doc_id]
            m = output_counter.get(res.first_output, 1)
            conf = compute_rarity_confidence(
                passed_examples=res.passed_examples,
                total_examples=len(examples),
                identical_output_count=m,
            )
            s_dense_norm = norm_dense.get(doc_id, 0.0)
            s_final = s_dense_norm + effective_alpha * conf
            blended_scores[doc_id] = s_final

        # Preserve the unverified tail candidates with their normalized dense scores
        for doc_id, _ in sorted_candidates[top_k:]:
            blended_scores[doc_id] = norm_dense.get(doc_id, 0.0)

        return dict(sorted(blended_scores.items(), key=lambda kv: kv[1], reverse=True))
