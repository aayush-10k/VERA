"""Unit tests for vera.chassis (preprocessing, dense chassis plumbing, negative mining)."""

from pathlib import Path

import numpy as np
import pytest

from vera.chassis.baseline import DenseChassis
from vera.chassis.mine_negatives import HardNegativeMiner
from vera.chassis.preprocess import ChassisPreprocessor, preprocess_query, strip_corpus_boilerplate


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
    q1 = preprocess_query(raw_query, remove_examples=False)
    assert "maximum subarray sum" in q1 and "Example 1" in q1
    q2 = preprocess_query(raw_query, remove_examples=True)
    assert "maximum subarray sum" in q2


def test_preprocessor_keeps_raw_and_stripped():
    pre = ChassisPreprocessor()
    stripped = pre.process_corpus({"d1": "import sys\nprint(1)"})
    assert pre.raw_corpus["d1"].startswith("import sys")
    assert "import sys" not in stripped["d1"]


def test_tfidf_chassis_indexing_and_search(tmp_path: Path):
    chassis = DenseChassis(backend="tfidf", cache_dir=tmp_path)
    corpus = {
        "d1": "def binary_search(arr, x): pass",
        "d2": "def quick_sort(arr): pass",
        "d3": "def dijkstra(graph, start): pass",
    }
    chassis.index_corpus(corpus)
    results = chassis.search({"q1": "dijkstra graph shortest path"}, top_k=2)
    assert list(results) == ["q1"] and len(results["q1"]) == 2
    assert next(iter(results["q1"])) == "d3"


def test_topk_from_matrix_orders_by_score():
    chassis = DenseChassis(backend="tfidf", cache_dir=Path("/tmp"))
    chassis.doc_ids = ["a", "b", "c"]
    sims = np.array([[0.1, 0.9, 0.5]], dtype=np.float32)
    out = chassis.topk_from_matrix(["q"], sims, top_k=2)
    assert list(out["q"]) == ["b", "c"]


def test_st_backend_never_falls_back_silently(tmp_path: Path):
    chassis = DenseChassis(model_name="definitely/not-a-real-model-xyz", backend="st", cache_dir=tmp_path)
    with pytest.raises(Exception):
        chassis.index_corpus({"d1": "print(1)"})


def test_hard_negative_miner():
    miner = HardNegativeMiner(top_k_dense=2)
    negatives = miner.mine_dense_negatives({"q1": "query text"}, {"q1": {"d1": 1}},
                                           {"q1": {"d1": 0.95, "d2": 0.88, "d3": 0.82, "d4": 0.40}})
    assert negatives["q1"] == ["d2", "d3"]
