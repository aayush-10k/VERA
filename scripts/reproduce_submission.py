"""
One-command reproduction of the submission JSON from a clean clone (CPU only)
============================================================================

    pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu
    python scripts/reproduce_submission.py            # highest rung whose dev fit is recorded in docs/
    python scripts/reproduce_submission.py --rung R0  # dense baseline only

What happens:
1. The AppsRetrieval parquet files are downloaded (HF hub, ~9 MB) and the model
   ``Alibaba-NLP/gte-modernbert-base`` (~300 MB) is fetched on first use.
2. Corpus (8,765) and test queries (3,765) are embedded once and cached under
   ``vera/chassis/cache`` (about 45 min + 25 min on a 4-core CPU; seconds afterwards).
3. The rung is evaluated through ``mteb.evaluate`` with the settings recorded on the dev
   split (``docs/dev_r2.json`` etc.); nothing is fit on test.
4. ``appsretrieval_results.json`` is written at the repository root and validated.

Expected wall time on 4 cores: R0 ≈ 75 min cold / 2 min warm; R2 adds ≈ 30 min of sandboxed
execution (3,765 queries × 150 candidates). Set ``--workers`` to your core count.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS = PROJECT_ROOT / "docs"


def highest_available_rung() -> str:
    if (DOCS / "dev_r4.json").exists() and json.loads((DOCS / "dev_r4.json").read_text()).get("decision") == "ADOPT":
        return "R4"
    if (DOCS / "dev_r3.json").exists() and json.loads((DOCS / "dev_r3.json").read_text()).get("decision") == "ADOPT":
        return "R3"
    if (DOCS / "dev_r2.json").exists():
        return "R2"
    return "R0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", choices=["R0", "R2", "R3", "R4"], default=None)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    rung = args.rung or highest_available_rung()
    py = sys.executable
    common = ["--promote"]
    workers = ["--workers", str(args.workers)] if args.workers else []
    if rung == "R0":
        cmd = [py, "scripts/m3_baseline.py", "--no-dev", *common]
    elif rung == "R2":
        alpha = json.loads((DOCS / "dev_r2.json").read_text())["selected_alpha"]
        cmd = [py, "scripts/m7_topk_verify.py", "--alpha", str(alpha), *workers, *common]
    elif rung == "R3":
        cmd = [py, "scripts/m8_corpus_gate.py", "--force-test", *workers, *common]
    else:
        cmd = [py, "scripts/m9_qbnorm.py", "--force-test", *workers, *common]
    print(f"[reproduce] rung {rung}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    from vera.mtebio.serializer import load_result, validate_mteb_result_schema

    data = load_result(PROJECT_ROOT / "appsretrieval_results.json")
    ok, errors = validate_mteb_result_schema(data)
    row = data["scores"]["test"][0]
    print(f"[reproduce] {'VALID' if ok else 'INVALID ' + str(errors)}  NDCG@10={row['ndcg_at_10']:.4f}  -> appsretrieval_results.json")


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    main()
