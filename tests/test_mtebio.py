"""Unit tests for vera.mtebio package."""

import json
from pathlib import Path
import pytest

from vera.mtebio.search_protocol import VERASearchProtocol
from vera.mtebio.encoder_fallback import VERAAbsEncoder
from vera.mtebio.serializer import (
    VERAJSONEncoder,
    save_mteb_task_result,
    validate_mteb_result_schema,
)


def test_search_protocol_interface():
    model = VERASearchProtocol(name="VERA-Test-Model")
    assert hasattr(model, "index")
    assert hasattr(model, "search")
    assert hasattr(model, "mteb_model_meta")

    # Index small corpus
    corpus = {
        "doc1": {"text": "def solve(): return 42"},
        "doc2": {"text": "def hello(): print('world')"},
    }
    model.index(corpus)
    assert len(model.corpus_index) == 2

    # Search query
    queries = {"q1": "return 42"}
    results = model.search(queries, top_k=2)
    assert "q1" in results
    assert "doc1" in results["q1"]


def test_abs_encoder_interface():
    encoder = VERAAbsEncoder()
    assert hasattr(encoder, "encode")
    assert hasattr(encoder, "encode_queries")
    assert hasattr(encoder, "encode_corpus")

    vecs = encoder.encode(["def foo(): pass", "def bar(): pass"])
    assert vecs.shape == (2, 768)


def test_serializer_and_schema_validation(tmp_path: Path):
    dummy_result = {
        "dataset_revision": "CoIR-APPS-v1.0",
        "mteb_dataset_name": "AppsRetrieval",
        "mteb_version": "2.0.1",
        "test": {
            "ndcg_at_10": 0.58,
            "mrr_at_10": 0.62,
            "map_at_10": 0.55,
            "recall_at_10": 0.70,
        },
    }

    out_file = tmp_path / "test_result.json"
    save_mteb_task_result(dummy_result, out_file)
    assert out_file.exists()

    with open(out_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    is_valid, errors = validate_mteb_result_schema(loaded)
    assert is_valid, f"Schema validation failed: {errors}"


def test_schema_validation_catches_invalid():
    invalid_result = {
        "mteb_dataset_name": "AppsRetrieval",
        "test": {},  # empty
    }
    is_valid, errors = validate_mteb_result_schema(invalid_result)
    assert not is_valid
    assert len(errors) > 0
