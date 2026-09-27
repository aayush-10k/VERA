"""
VERA Result Serializer & Schema Validator
==========================================
Handles robust JSON serialization of MTEB TaskResult objects.
Resolves known serialization quirks (datetime objects, numpy types)
and validates output schemas against hackathon submission requirements.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class VERAJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder handling datetime, sets, and numpy scalars."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, set):
            return sorted(list(obj))
        # Handle numpy types if present without hard dependency
        type_str = str(type(obj))
        if "numpy" in type_str:
            if hasattr(obj, "item"):
                return obj.item()
            if hasattr(obj, "tolist"):
                return obj.tolist()
        return super().default(obj)


def save_mteb_task_result(
    task_result_dict: Dict[str, Any],
    output_path: Path,
    indent: int = 2,
) -> Path:
    """
    Saves a task result dictionary to disk with clean formatting
    and ISO datetime serialization.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(task_result_dict, f, indent=indent, cls=VERAJSONEncoder)

    print(f"[VERA Serializer] Successfully saved task result to {output_path} ({output_path.stat().st_size / 1024:.1f} KB).")
    return output_path


def validate_mteb_result_schema(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validates that the generated JSON satisfies the MTEB TaskResult schema
    and contains all necessary metrics for the AppsRetrieval evaluation.
    """
    errors: List[str] = []

    # Check top-level keys
    required_top = ["dataset_revision", "mteb_dataset_name", "test"]
    for k in required_top:
        if k not in data and not any(k in key for key in data.keys()):
            errors.append(f"Missing required top-level key: {k}")

    # Check test split metrics
    test_metrics = data.get("test", {})
    if not isinstance(test_metrics, dict) or len(test_metrics) == 0:
        errors.append("Empty or missing 'test' metrics split in TaskResult.")
    else:
        # Check primary benchmark metrics
        essential_metrics = [
            "ndcg_at_10",
            "mrr_at_10",
            "map_at_10",
            "recall_at_10",
        ]
        for m in essential_metrics:
            if m not in test_metrics:
                errors.append(f"Missing essential metric '{m}' under test split.")

    is_valid = len(errors) == 0
    return is_valid, errors
