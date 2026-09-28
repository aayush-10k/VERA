"""Unit tests for vera.mtebio (SearchProtocol wiring, TaskResult serialization/validation)."""

import json
from pathlib import Path

import pytest

from vera.chassis.baseline import DenseChassis
from vera.mtebio.search_protocol import VERASearchProtocol, _ids_and_texts
from vera.mtebio.serializer import save_mteb_task_result, validate_mteb_result_schema


CORPUS = {
    "d1": "n = int(input())\nprint(n * n)",
    "d2": "s = input()\nprint(s[::-1])",
    "d3": "a, b = map(int, input().split())\nprint(a + b)",
}


def _protocol(tmp_path: Path) -> VERASearchProtocol:
    return VERASearchProtocol(chassis=DenseChassis(backend="tfidf", cache_dir=tmp_path), name="vera/test-model")


def test_ids_and_texts_accepts_all_container_shapes():
    assert _ids_and_texts({"a": "x", "b": {"text": "y"}}) == (["a", "b"], ["x", "y"])
    assert _ids_and_texts([{"id": "a", "text": "x"}, {"_id": "b", "text": "y"}]) == (["a", "b"], ["x", "y"])


def test_search_protocol_index_and_search(tmp_path: Path):
    model = _protocol(tmp_path)
    assert hasattr(model, "index") and hasattr(model, "search") and hasattr(model, "mteb_model_meta")
    assert model.mteb_model_meta.name == "vera/test-model"

    model.index(CORPUS)
    assert set(model.raw_corpus) == set(CORPUS)

    results = model.search({"q1": "reverse the string s and print it"}, top_k=2)
    assert list(results) == ["q1"]
    assert len(results["q1"]) == 2
    assert next(iter(results["q1"])) == "d2"  # lexical match ranks the reversal snippet first


def test_search_protocol_is_recognised_by_mteb(tmp_path: Path):
    mteb = pytest.importorskip("mteb")
    from mteb.models.models_protocols import SearchProtocol

    assert isinstance(_protocol(tmp_path), SearchProtocol)


def test_serializer_and_schema_validation(tmp_path: Path):
    task_result = {
        "dataset_revision": "f22508f96b7a36c2415181ed8bb76f76e04ae2d5",
        "task_name": "AppsRetrieval",
        "mteb_version": "2.21.8",
        "evaluation_time": 12.3,
        "kg_co2_emissions": None,
        "scores": {"test": [{
            "ndcg_at_10": 0.58, "mrr_at_10": 0.62, "map_at_10": 0.55, "recall_at_10": 0.70,
            "main_score": 0.58, "hf_subset": "default", "languages": ["eng-Latn", "python-Code"],
        }]},
    }
    out = tmp_path / "r.json"
    save_mteb_task_result(task_result, out, extra={"rung": "R0"})
    loaded = json.loads(out.read_text())
    ok, errors = validate_mteb_result_schema(loaded)
    assert ok, errors
    assert loaded["vera_run"]["rung"] == "R0"


def test_schema_validation_catches_legacy_and_invalid_layouts():
    legacy = {"dataset_revision": "x", "mteb_dataset_name": "AppsRetrieval", "test": {"ndcg_at_10": 0.5}}
    ok, errors = validate_mteb_result_schema(legacy)
    assert not ok and errors

    bad_range = {
        "dataset_revision": "x", "task_name": "AppsRetrieval", "mteb_version": "2", "evaluation_time": 1,
        "scores": {"test": [{"ndcg_at_10": 1.5, "mrr_at_10": 0, "map_at_10": 0, "recall_at_10": 0, "main_score": 0,
                             "hf_subset": "default", "languages": []}]},
    }
    ok, errors = validate_mteb_result_schema(bad_range)
    assert not ok and any("out of range" in e for e in errors)
