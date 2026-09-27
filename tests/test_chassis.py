"""Unit tests for vera.chassis package."""

import numpy as np
import pytest
from pathlib import Path

from vera.chassis.preprocess import strip_corpus_boilerplate, preprocess_query, ChassisPreprocessor
from vera.chassis.baseline import DenseChassis
from vera.chassis.mine_negatives import HardNegativeMiner


def test_strip_corpus_boilerplate():
    raw_code = """import sys
input = sys.stdin.readline
sys.setrecursionlimit(2000)

def solve():
    n = int(input())
    return n * 2

if __name__ == '__main__':
    print(solve())
"""
    cleaned = strip_corpus_boilerplate(raw_code)
    assert "sys.stdin.readline" not in cleaned
    assert "setrecursionlimit" not in cleaned
    assert "def solve():" in cleaned
    assert "return n * 2" in cleaned


def test_preprocess_query():
    raw_query = "Find the maximum subarray sum.\n\nExample 1:\nInput: [1, 2, 3]\nOutput: 6"
    
    # Normal query
    q1 = preprocess_query(raw_query, remove_examples=False)
    assert "maximum subarray sum" in q1
    assert "Example 1" in q1

    # Query with examples removed
    q2 = preprocess_query(raw_query, remove_examples=True)
    assert "maximum subarray sum" in q2


def test_dense_chassis_indexing_and_search(tmp_path: Path):
    chassis = DenseChassis(cache_dir=tmp_path, use_fallback=True)

    corpus = {
        "d1": "def binary_search(arr, x): pass",
        "d2": "def quick_sort(arr): pass",
        "d3": "def dijkstra(graph, start): pass",
    }
    chassis.index_corpus(corpus)
    assert chassis.corpus_embeddings is not None
    assert chassis.corpus_embeddings.shape[0] == 3

    # Search
    queries = {"q1": "graph shortest path algorithm"}
    results = chassis.search(queries, top_k=2)
    assert "q1" in results
    assert len(results["q1"]) == 2


def test_hard_negative_miner():
    miner = HardNegativeMiner(top_k_dense=2)
    train_queries = {"q1": "query text"}
    train_qrels = {"q1": {"d1": 1}}
    candidate_scores = {
        "q1": {"d1": 0.95, "d2": 0.88, "d3": 0.82, "d4": 0.40}
    }

    negatives = miner.mine_dense_negatives(train_queries, train_qrels, candidate_scores)
    assert "q1" in negatives
    assert "d1" not in negatives["q1"]  # Gold must NOT be a negative
    assert negatives["q1"] == ["d2", "d3"]
