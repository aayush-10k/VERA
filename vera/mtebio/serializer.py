"""
VERA Result Serializer & Schema Validator
==========================================
Persists MTEB ``TaskResult`` objects and validates the on-disk JSON against the
shape MTEB itself writes::

    {
      "dataset_revision": "...", "task_name": "AppsRetrieval", "mteb_version": "2.x",
      "scores": {"test": [{"ndcg_at_10": ..., "main_score": ..., "hf_subset": "default", "languages": [...]}]},
      "evaluation_time": ..., "kg_co2_emissions": null
    }
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

ESSENTIAL_METRICS = ("ndcg_at_10", "mrr_at_10", "map_at_10", "recall_at_10", "main_score")


class VERAJSONEncoder(json.JSONEncoder):
    """JSON encoder handling datetime, sets and numpy scalars."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, set):
            return sorted(obj)
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return super().default(obj)


def task_result_to_dict(task_result: Any) -> Dict[str, Any]:
    """``TaskResult`` (mteb) or plain dict -> plain dict."""
    if isinstance(task_result, dict):
        return task_result
    if hasattr(task_result, "to_dict"):
        return task_result.to_dict()
    if hasattr(task_result, "model_dump"):
        return task_result.model_dump(mode="json")
    raise TypeError(f"Cannot serialize {type(task_result)}")


def save_mteb_task_result(task_result: Any, output_path: Path, indent: int = 2, extra: Dict[str, Any] | None = None) -> Path:
    """Write the TaskResult JSON. ``extra`` (e.g. VERA run settings) is stored under ``vera_run`` without touching MTEB keys."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(task_result_to_dict(task_result))
    if extra:
        data["vera_run"] = extra
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, cls=VERAJSONEncoder)
    print(f"[VERA Serializer] Saved task result -> {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    return output_path


def validate_mteb_result_schema(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate the MTEB TaskResult layout for AppsRetrieval (test split, essential metrics present and in [0, 1])."""
    errors: List[str] = []
    for k in ("dataset_revision", "task_name", "mteb_version", "scores", "evaluation_time"):
        if k not in data:
            errors.append(f"Missing required top-level key: {k}")
    if data.get("task_name") not in (None, "AppsRetrieval"):
        errors.append(f"task_name is {data.get('task_name')!r}, expected 'AppsRetrieval'")
    scores = data.get("scores", {})
    test_rows = scores.get("test") if isinstance(scores, dict) else None
    if not isinstance(test_rows, list) or not test_rows:
        errors.append("scores.test must be a non-empty list")
        return False, errors
    row = test_rows[0]
    for m in ESSENTIAL_METRICS:
        if m not in row:
            errors.append(f"Missing metric '{m}' in scores.test[0]")
        elif not isinstance(row[m], (int, float)) or not (0.0 <= float(row[m]) <= 1.0):
            errors.append(f"Metric '{m}' out of range: {row[m]!r}")
    for k in ("hf_subset", "languages"):
        if k not in row:
            errors.append(f"Missing '{k}' in scores.test[0]")
    return (not errors), errors


def load_result(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main_score(path: Path) -> float:
    return float(load_result(path)["scores"]["test"][0]["main_score"])
