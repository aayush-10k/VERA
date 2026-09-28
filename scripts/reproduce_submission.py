"""
One-command reproduction of the submission JSON from a clean clone (CPU only)
============================================================================

    pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu
    python scripts/reproduce_submission.py              # the submitted pipeline (R4: gated extension + QB-Norm)
    python scripts/reproduce_submission.py --dry-run    # print the milestone command that would run
    python scripts/reproduce_submission.py --rung R2    # any lower rung of the ship ladder

Rungs (``docs/ablations.md`` rows) and the recorded dev fit each one applies to the test split:

    R0   dense chassis only                                  scripts/m3_baseline.py
    R2   + top-150 verification, alpha from docs/dev_r2.json  scripts/m7_topk_verify.py --alpha
    R3   + gated extension, tau from docs/dev_r3.json         scripts/m8_corpus_gate.py --tau
    R4a  R2 + QB-Norm (k, beta from docs/dev_r4.json)         scripts/m9_qbnorm.py
    R4   R3 + QB-Norm, tau from docs/dev_r4_final.json        scripts/m8_corpus_gate.py --qbnorm --tau   <- submitted

Nothing is fit on test: the dev-selected alpha / (k, beta) / tau recorded under ``docs/`` are
passed in unchanged. ``--promote`` overwrites ``appsretrieval_results.json`` at the repository root.

What happens on a clean clone (4-core CPU, measured):
1. AppsRetrieval parquet files (~9 MB) and ``Alibaba-NLP/gte-modernbert-base`` (~300 MB) are
   downloaded on first use.
2. Corpus (8,765 docs) and queries are embedded once and cached under ``vera/chassis/cache``:
   about 45 min (corpus) + 25 min (test queries) + a few minutes (train queries, for QB-Norm).
3. Verification runs in the process-isolated sandbox: the base top-150 pass costs about 57 min
   (3,765 x 150 candidates); R4 adds about 60 min for the gated extension to dense rank 500
   (2,038 queries extended, 479k extra executions). Per-query results are cached, so a re-run
   of the same rung (or another rung sharing the base pass) takes minutes.
4. The rung's ``TaskResult`` JSON is written under ``artifacts/`` and copied to
   ``appsretrieval_results.json``, then validated and compared with the recorded number.

Expected NDCG@10: R0 0.5683 · R2 0.8712 · R3 0.8822 · R4a 0.8772 · R4 0.8874.
Total cold wall time for R4 is therefore about 3 h; warm re-runs a few minutes.
The sandbox uses ``os.fork`` and ``resource``: Linux or macOS (WSL2 on Windows).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS = PROJECT_ROOT / "docs"
ARTIFACTS = PROJECT_ROOT / "artifacts"

RUNGS = ("R0", "R2", "R3", "R4a", "R4")
RECORDED_ARTIFACT = {
    "R0": "m3_r0_results.json",
    "R2": "m7_r2_results.json",
    "R3": "m8_r3_results.json",
    "R4a": "m9_r4_results.json",
    "R4": "m8_r4_final_results.json",
}


def _dev(name: str) -> dict:
    path = DOCS / name
    return json.loads(path.read_text()) if path.exists() else {}


def _adopted(name: str) -> bool:
    return _dev(name).get("decision") == "ADOPT"


def highest_available_rung() -> str:
    """Highest rung whose dev fit is recorded under docs/ with an ADOPT decision."""
    if _adopted("dev_r4_final.json"):
        return "R4"
    if _adopted("dev_r4.json"):
        return "R4a"
    if _adopted("dev_r3.json"):
        return "R3"
    if (DOCS / "dev_r2.json").exists():
        return "R2"
    return "R0"


def command_for(rung: str, python: str, workers: int | None) -> list[str]:
    w = ["--workers", str(workers)] if workers else []
    if rung == "R0":
        return [python, "scripts/m3_baseline.py", "--no-dev", "--promote"]
    if rung == "R2":
        alpha = _dev("dev_r2.json")["selected_alpha"]
        return [python, "scripts/m7_topk_verify.py", "--alpha", str(alpha), *w, "--promote"]
    if rung == "R3":
        tau = _dev("dev_r3.json")["best_tau"]
        return [python, "scripts/m8_corpus_gate.py", "--tau", str(tau), *w, "--promote"]
    if rung == "R4a":
        return [python, "scripts/m9_qbnorm.py", "--force-test", *w, "--promote"]
    if rung == "R4":
        tau = _dev("dev_r4_final.json")["best_tau"]
        return [python, "scripts/m8_corpus_gate.py", "--qbnorm", "--tau", str(tau), *w, "--promote"]
    raise ValueError(rung)


def recorded_ndcg10(rung: str) -> float | None:
    path = ARTIFACTS / RECORDED_ARTIFACT[rung]
    if not path.exists():
        return None
    return float(json.loads(path.read_text())["scores"]["test"][0]["ndcg_at_10"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rung", choices=RUNGS, default=None, help="default: highest rung with a recorded dev fit (R4)")
    ap.add_argument("--workers", type=int, default=None, help="sandbox workers (default: CPU count)")
    ap.add_argument("--dry-run", action="store_true", help="print the milestone command and exit")
    args = ap.parse_args()
    rung = args.rung or highest_available_rung()
    cmd = command_for(rung, sys.executable, args.workers)
    recorded = recorded_ndcg10(rung)
    print(f"[reproduce] rung {rung}: {' '.join(cmd)}", flush=True)
    if recorded is not None:
        print(f"[reproduce] recorded test NDCG@10 for {rung}: {recorded:.4f} (artifacts/{RECORDED_ARTIFACT[rung]})", flush=True)
    if args.dry_run:
        return
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

    from vera.mtebio.serializer import load_result, validate_mteb_result_schema

    data = load_result(PROJECT_ROOT / "appsretrieval_results.json")
    ok, errors = validate_mteb_result_schema(data)
    ndcg = float(data["scores"]["test"][0]["ndcg_at_10"])
    status = "VALID" if ok else f"INVALID {errors}"
    print(f"[reproduce] {status}  NDCG@10={ndcg:.4f}  -> appsretrieval_results.json")
    if recorded is not None:
        delta = ndcg - recorded
        verdict = "matches" if abs(delta) <= 0.002 else "DIFFERS FROM"
        print(f"[reproduce] {verdict} the recorded {rung} number {recorded:.4f} ({delta:+.4f})")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    main()
