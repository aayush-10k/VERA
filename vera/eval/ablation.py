"""
Ablation table generator -> docs/ablations.md
=============================================
One row per rung actually run. Test numbers come from the MTEB ``TaskResult`` JSONs in
``artifacts/``; dev numbers from ``docs/dev_*.json``. A rung without an artifact is listed
as *not run* — no row is ever filled by hand. This table is PPT slide 5, verbatim.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = PROJECT_ROOT / "artifacts"
DOCS = PROJECT_ROOT / "docs"

# (label, test artifact, dev json, description)
ROWS: List[Tuple[str, str, Optional[str], str]] = [
    ("REF  lexical TF-IDF (word+bigram)", "ref_tfidf_results.json", "dev_ref_tfidf.json", "reference only, never submitted"),
    ("R0   gte-modernbert-base, 8192 ctx, stripped corpus", "m3_r0_results.json", "dev_r0.json", "dense chassis, zero-shot"),
    ("R0b  R0 with raw (unstripped) corpus", "m3_r0_raw_results.json", None, "preprocessing ablation"),
    ("R0c  R0 with the example section dropped from the embedded query", "m3_r0_noqex_results.json", None, "query ablation"),
    ("R1   R0 + LoRA fine-tune", "m5_r1_results.json", "dev_r1.json", "requires GPU session"),
    ("R2   R0/R1 + top-150 verification, rarity boost", "m7_r2_results.json", "dev_r2.json", "alpha fit on dev"),
    ("R3   R2 + corpus-wide verification behind gate", "m8_r3_results.json", "dev_r3.json", "router tau fit on dev"),
    ("R4   + QB-Norm demotion", "m9_r4_results.json", "dev_r4.json", "beta fit on dev"),
]


def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _test_row(data: Dict[str, Any]) -> Dict[str, Any]:
    row = data["scores"]["test"][0]
    run = data.get("vera_run", {})
    wall = run.get("verification_wall_s_uncached") or run.get("wall_time_s") or data.get("evaluation_time")
    return {"ndcg10": row["ndcg_at_10"], "mrr10": row["mrr_at_10"], "r10": row["recall_at_10"], "r100": row["recall_at_100"],
            "wall": wall, "settings": {k: run[k] for k in ("alpha", "top_k", "beta", "tau", "model") if k in run}}


def _dev_ndcg(dev: Optional[Dict[str, Any]]) -> Optional[float]:
    if not dev:
        return None
    for key in ("dev_r4", "dev_r3", "dev_r2", "dev_r1", "dev"):  # most specific first: dev_rN.json also carries the R2 reference
        if key in dev and isinstance(dev[key], dict) and "ndcg_at_10" in dev[key]:
            return dev[key]["ndcg_at_10"]
    return None


def _fmt(x: Optional[float], pct: bool = True) -> str:
    if x is None:
        return "—"
    return f"{100 * x:.2f}" if pct else f"{x:.4f}"


def build_table() -> str:
    lines = ["# Ablations — AppsRetrieval (CoIR), every row measured", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} by `vera/eval/ablation.py` from the MTEB "
             "`TaskResult` JSONs in `artifacts/` (test split, 3,765 queries, scored by `mteb.evaluate`) and the dev-split "
             "JSONs in `docs/` (500 held-out train pairs). Δ is versus the previous *measured* rung.", "",
             "| Rung | Test NDCG@10 | Δ | Test MRR@10 | Test R@10 | Test R@100 | Dev NDCG@10 | Wall time | Notes |",
             "|---|---|---|---|---|---|---|---|---|"]
    prev: Optional[float] = None
    for label, art, dev_name, desc in ROWS:
        data = _load(ARTIFACTS / art)
        dev = _load(DOCS / dev_name) if dev_name else None
        if data is None:
            lines.append(f"| {label} | *not run* | | | | | {_fmt(_dev_ndcg(dev))} | | {desc} |")
            continue
        t = _test_row(data)
        delta = "" if prev is None or label.startswith("REF") else f"{100 * (t['ndcg10'] - prev):+.2f}"
        wall = f"{t['wall'] / 60:.0f} min" if isinstance(t["wall"], (int, float)) and t["wall"] >= 120 else (f"{t['wall']:.0f} s" if isinstance(t["wall"], (int, float)) else "—")
        settings = ", ".join(f"{k}={v}" for k, v in t["settings"].items() if k != "model")
        lines.append(f"| {label} | **{_fmt(t['ndcg10'])}** | {delta} | {_fmt(t['mrr10'])} | {_fmt(t['r10'])} | {_fmt(t['r100'])} | "
                     f"{_fmt(_dev_ndcg(dev))} | {wall} | {desc}{'; ' + settings if settings else ''} |")
        if not label.startswith("REF"):
            prev = t["ndcg10"]
    lines += ["", "Reference points from the field (not ours): BM25 ≈ 0.95 NDCG@10 (CtrlFind), gte-modernbert-base "
                  "zero-shot 56.4 (Granite-R2 paper, 1024-token cap) / 57.5 reproduced by a PRISM competitor.", ""]
    dev_r2 = _load(DOCS / "dev_r2.json")
    if dev_r2 and "alpha_grid" in dev_r2:
        lines += ["## R2 alpha sweep (dev NDCG@10)", "", "| alpha | " + " | ".join(dev_r2["alpha_grid"].keys()) + " |",
                  "|---|" + "---|" * len(dev_r2["alpha_grid"]),
                  "| NDCG@10 | " + " | ".join(f"{100 * v['ndcg_at_10']:.2f}" for v in dev_r2["alpha_grid"].values()) + " |", ""]
        d = dev_r2.get("diagnostics", {})
        if d:
            lines += [f"Dev diagnostics: gold inside the dense top-{dev_r2.get('top_k')} for {d.get('gold_in_dense_topk')}/{d.get('dev_queries')} "
                      f"queries; gold passes its own examples for {d.get('gold_passes_own_examples')}; "
                      f"exactly one full passer in {d.get('queries_with_exactly_one_full_passer')} queries; "
                      f"mean full passers per query {d.get('mean_candidates_passing_all')}.", ""]
    return "\n".join(lines)


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / "ablations.md"
    out.write_text(build_table(), encoding="utf-8")
    print(build_table())
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
