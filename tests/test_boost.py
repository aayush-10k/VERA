"""
Unit Tests for VERA Top-K Verification Re-Ranking & Rarity Boost (R2)
====================================================================
"""

import math
import pytest
from vera.verify.boost import (
    TopKVerifier,
    compute_rarity_confidence,
    min_max_normalize,
)
from vera.verify.executor import VerificationSandbox
from vera.verify.parser import ExamplePair, WorkedExampleParser


class TestRarityBoost:
    def test_compute_rarity_confidence_edge_cases(self):
        # 0 total or passed
        assert compute_rarity_confidence(0, 5, 1) == 0.0
        assert compute_rarity_confidence(2, 0, 1) == 0.0

        # Unique passing candidate (m=1)
        conf_unique = compute_rarity_confidence(2, 2, 1)
        assert conf_unique == 1.0

        # Non-unique passing candidates (m > 1) -> rarity discounted
        conf_m3 = compute_rarity_confidence(2, 2, 3)
        expected_m3 = 1.0 / (1.0 + math.log2(3))
        assert abs(conf_m3 - expected_m3) < 1e-5
        assert conf_m3 < conf_unique

        # Higher m -> lower confidence
        conf_m15 = compute_rarity_confidence(2, 2, 15)
        assert conf_m15 < conf_m3

        # Partial passes
        conf_half = compute_rarity_confidence(1, 2, 1)
        assert conf_half == 0.5

    def test_min_max_normalize(self):
        assert min_max_normalize({}) == {}
        # Identical values
        res_equal = min_max_normalize({"a": 5.0, "b": 5.0})
        assert res_equal == {"a": 1.0, "b": 1.0}

        # Scaled values
        scores = {"doc1": 10.0, "doc2": 20.0, "doc3": 30.0}
        norm = min_max_normalize(scores)
        assert norm["doc1"] == 0.0
        assert norm["doc2"] == 0.5
        assert norm["doc3"] == 1.0


class TestTopKVerifier:
    def test_passing_candidate_boosted_over_failing_candidate(self):
        verifier = TopKVerifier(default_alpha=0.35)

        query_text = """
Given an integer n, print its square.
-----Sample Input-----
4
-----Sample Output-----
16
"""
        # doc_fail has higher initial dense score (0.90) but wrong code
        # doc_pass has lower initial dense score (0.80) but correct code
        dense_scores = {
            "doc_fail": 0.90,
            "doc_pass": 0.80,
            "doc_tail": 0.20,
        }
        corpus_dict = {
            "doc_fail": "x = int(input())\nprint(x + 10)",  # prints 14 != 16
            "doc_pass": "x = int(input())\nprint(x * x)",   # prints 16 == 16
            "doc_tail": "print(0)",
        }

        reranked = verifier.rerank_query(
            query_text=query_text,
            dense_scores=dense_scores,
            corpus_dict=corpus_dict,
            top_k=2,
            alpha=0.35,
        )

        ranked_docs = list(reranked.keys())
        # The correct solution should be promoted to #1!
        assert ranked_docs[0] == "doc_pass"
        assert reranked["doc_pass"] > reranked["doc_fail"]

    def test_rerank_query_without_examples_preserves_order(self):
        verifier = TopKVerifier(default_alpha=0.20)
        query_text = "Compute the shortest path in a weighted graph."  # No sample I/O

        dense_scores = {
            "doc1": 0.95,
            "doc2": 0.85,
            "doc3": 0.70,
        }
        corpus_dict = {
            "doc1": "pass",
            "doc2": "pass",
            "doc3": "pass",
        }

        reranked = verifier.rerank_query(
            query_text=query_text,
            dense_scores=dense_scores,
            corpus_dict=corpus_dict,
            top_k=3,
        )

        ranked_docs = list(reranked.keys())
        assert ranked_docs == ["doc1", "doc2", "doc3"]
