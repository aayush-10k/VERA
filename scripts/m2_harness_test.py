"""
VERA Milestone 2 — MTEB harness proof-of-life
=============================================
Proves the submission mechanics with the *real* MTEB harness before anything clever
exists: ``mteb.evaluate(VERASearchProtocol(...), AppsRetrieval)`` with the cheap TF-IDF
backend, then validates the emitted ``TaskResult`` JSON.

Also serves as the lexical reference row (``artifacts/ref_tfidf_results.json``) for
``docs/ablations.md``. Never promoted to ``appsretrieval_results.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vera.chassis.baseline import DenseChassis
from vera.mtebio.search_protocol import VERASearchProtocol
from vera.mtebio.serializer import save_mteb_task_result, validate_mteb_result_schema

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PROOF_OF_LIFE_JSON = ARTIFACTS_DIR / "m2_proof_of_life.json"
REF_TFIDF_JSON = ARTIFACTS_DIR / "ref_tfidf_results.json"


def main() -> None:
    import mteb

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 66)
    print(" VERA M2 — MTEB harness proof-of-life (TF-IDF backend)")
    print("=" * 66)
    chassis = DenseChassis(backend="tfidf")
    model = VERASearchProtocol(chassis=chassis, name="vera/VERA-ProofOfLife-tfidf")
    task = mteb.get_task("AppsRetrieval")
    t0 = time.time()
    result = mteb.evaluate(model, task, cache=None, overwrite_strategy="always", co2_tracker=False)
    tr = result.task_results[0]
    extra = {"rung": "REF", "backend": "tfidf", "wall_time_s": round(time.time() - t0, 1), "timings": model.timings}
    save_mteb_task_result(tr, PROOF_OF_LIFE_JSON, extra=extra)
    save_mteb_task_result(tr, REF_TFIDF_JSON, extra=extra)

    data = json.loads(PROOF_OF_LIFE_JSON.read_text())
    ok, errors = validate_mteb_result_schema(data)
    row = data["scores"]["test"][0]
    print(f"\n task={data['task_name']} rev={data['dataset_revision']} mteb={data['mteb_version']}")
    print(f" NDCG@10={row['ndcg_at_10']:.4f} MRR@10={row['mrr_at_10']:.4f} R@100={row['recall_at_100']:.4f} ({extra['wall_time_s']}s)")
    if not ok:
        print(" [FAILED] schema:", errors)
        sys.exit(1)
    print(f" [PASSED] TaskResult schema valid -> {PROOF_OF_LIFE_JSON}")


if __name__ == "__main__":
    main()
