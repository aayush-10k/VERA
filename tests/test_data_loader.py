"""Unit tests for vera.data.loader."""

import pytest
from pathlib import Path
from vera.data.loader import AppsRetrievalDataset, DEV_SPLIT_FILE


def test_apps_dataset_loading_and_counts():
    ds = AppsRetrievalDataset()
    
    test_queries = ds.get_test_queries()
    corpus = ds.get_corpus()
    test_qrels = ds.get_test_qrels()
    train_qrels = ds.train_qrels

    # Verify counts according to competition rules & audit
    assert len(test_queries) == 3765, f"Expected 3765 test queries, got {len(test_queries)}"
    assert len(corpus) == 8765, f"Expected 8765 corpus documents, got {len(corpus)}"
    assert len(test_qrels) == 3765, f"Expected 3765 test qrel entries, got {len(test_qrels)}"
    assert len(train_qrels) == 5000, f"Expected 5000 train qrel entries, got {len(train_qrels)}"

    # Check that each test query has exactly 1 gold solution
    for qid, docs in test_qrels.items():
        assert len(docs) == 1, f"Query {qid} has {len(docs)} gold documents, expected 1"


def test_dev_split_partition():
    ds = AppsRetrievalDataset()
    split = ds.get_or_create_dev_split(dev_size=500, seed=42)

    assert DEV_SPLIT_FILE.exists()
    assert len(split["dev_query_ids"]) == 500
    assert len(split["train_query_ids"]) == 4500

    # Ensure zero overlap between dev and train partitions
    dev_set = set(split["dev_query_ids"])
    train_set = set(split["train_query_ids"])
    overlap = dev_set.intersection(train_set)
    assert len(overlap) == 0, f"Dev and train split have overlapping IDs: {overlap}"
