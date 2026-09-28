"""
VERA Milestone 8 — R3: gated corpus-wide verification (conditional on the M4 gate)
=================================================================================
The M4 gate passed (docs/goldrun.md: 86.7% projected), so this rung is evaluated.

Design (Plan.md 7.4): an **uncertainty router** skips the extension when the dense
top-1/top-2 margin is decisive; otherwise verification extends from the dense top-150 to
dense ranks 150..``extend_k`` (``--full-corpus`` = every gate-eligible document) filtered by
the **static signature gate** (program I/O shape must be compatible with the example's
input layout).

Dev protocol: every dev query's base top-K *and* extension candidates are verified once
(through the per-query verification cache shared with m7/m9, so nothing already executed
is re-run); each ``tau`` is then simulated by restricting the candidate set, rarity ``m``
recomputed over that set. ``--qbnorm`` applies the QB-Norm demotion selected in
``docs/dev_r4.json`` to the dense scores first (dev bank without the dev statements), so the
decision is made for the *combined* final pipeline R3 + QB-Norm (Plan.md R4).

If the extension does not beat the base pipeline on dev, the rejection is recorded and no
test JSON is produced unless ``--force-test``.

``--tau`` pins the router threshold to a value already selected on dev (``docs/dev_r3.json`` /
``docs/dev_r4_final.json``) and skips the dev extension verification and sweep entirely — the
reproduction path used by ``scripts/reproduce_submission.py`` (nothing is fit on test; the recorded
dev fit is applied as-is, exactly like ``m7_topk_verify.py --alpha``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vera.chassis.baseline import DenseChassis
from vera.chassis.preprocess import ChassisPreprocessor
from vera.chassis.qbnorm import QBNorm
from vera.data.loader import AppsRetrievalDataset
from vera.eval.metrics import compute_retrieval_metrics, rank_of_gold
from vera.gate.router import SignatureGate, UncertaintyRouter
from vera.mtebio.search_protocol import VERASearchProtocol
from vera.mtebio.serializer import save_mteb_task_result, validate_mteb_result_schema
from vera.verify.boost import GatedVerifier, QueryVerification, TopKVerifier, compute_rarity_confidence
from vera.verify.executor import VerificationSandbox

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DOCS_DIR = PROJECT_ROOT / "docs"
OFFICIAL_SUBMISSION_JSON = PROJECT_ROOT / "appsretrieval_results.json"
TAU_GRID = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 1.0]  # 0.0 == base pipeline only, 1.0 == extend every query


def load_dev_verification(path: Path) -> Dict[str, QueryVerification]:
    """Kept for backwards compatibility with earlier scripts (m9 imports it)."""
    raw = json.loads(path.read_text())
    out = {}
    for q, d in raw.items():
        qv = QueryVerification(n_examples=d["n_examples"])
        qv.conf, qv.passed, qv.m_first, qv.n_all_pass = d["conf"], d["passed"], d["m_first"], d["n_all_pass"]
        out[q] = qv
    return out


def restrict(qv: QueryVerification, ids: List[str]) -> QueryVerification:
    """Confidences over a candidate subset, rarity m recomputed over that subset."""
    out = QueryVerification(n_examples=qv.n_examples)
    if qv.n_examples == 0:
        return out
    out.m_first = sum(1 for d in ids if qv.passed.get(d, 0) > 0)
    for d in ids:
        p = qv.passed.get(d, 0)
        out.passed[d] = p
        out.conf[d] = compute_rarity_confidence(p, qv.n_examples, out.m_first if p > 0 else 1)
        if p == qv.n_examples:
            out.n_all_pass += 1
    return out


def main():
    ap = argparse.ArgumentParser(description="VERA R3 gated corpus-wide verification (optionally + QB-Norm = plan R4)")
    ap.add_argument("--model", default="Alibaba-NLP/gte-modernbert-base")
    ap.add_argument("--base-k", type=int, default=150)
    ap.add_argument("--extend-k", type=int, default=500)
    ap.add_argument("--full-corpus", action="store_true")
    ap.add_argument("--alpha", type=float, default=None, help="alpha (default: selected by m7 in docs/dev_r2.json)")
    ap.add_argument("--tau", type=float, default=None, help="fixed router tau (skips the dev extension verification + sweep; use the value recorded in docs/dev_r3.json or docs/dev_r4_final.json)")
    ap.add_argument("--qbnorm", action="store_true", help="apply the QB-Norm selected in docs/dev_r4.json (combined final pipeline)")
    ap.add_argument("--timeout", type=float, default=0.75)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--dev-only", action="store_true")
    ap.add_argument("--force-test", action="store_true")
    ap.add_argument("--promote", action="store_true")
    args = ap.parse_args()

    alpha = args.alpha if args.alpha is not None else float(json.loads((DOCS_DIR / "dev_r2.json").read_text())["selected_alpha"])
    label = "r4_final" if args.qbnorm else "r3"
    rung_label = "R4-final (R3 + QB-Norm)" if args.qbnorm else "R3"
    if args.tau is not None and args.dev_only:
        ap.error("--tau pins the dev fit; it cannot be combined with --dev-only")
    ds = AppsRetrievalDataset()
    pre = ChassisPreprocessor()
    raw_corpus = ds.get_corpus()
    chassis = DenseChassis(model_name=args.model, batch_size=8)
    chassis.index_corpus(pre.process_corpus(raw_corpus))
    short = args.model.split("/")[-1]
    cache_path = str(chassis.cache_dir / f"r2_verification_{short}_k{args.base_k}_t{args.timeout}.json")

    # ---- dev dense ranking (optionally QB-normalised with the dev bank) ------------------------------
    split = ds.get_or_create_dev_split()
    dev_qids = split["dev_query_ids"]
    dev_set = set(dev_qids)
    all_train = pre.process_queries({q: ds.queries_dict[q] for q in sorted(ds.train_qrels)})
    qids, embs = chassis.encode_queries(all_train, tag="train_queries")
    pos = {q: i for i, q in enumerate(qids)}
    sims = chassis.similarity(embs[[pos[q] for q in dev_qids]])
    qb_cfg = None
    if args.qbnorm:
        dev_r4 = json.loads((DOCS_DIR / "dev_r4.json").read_text())
        qb_cfg = {"k": int(dev_r4["best_k"]), "beta": float(dev_r4["best_beta"])}
        bank_dev = embs[[pos[q] for q in qids if q not in dev_set]]
        hub_dev = QBNorm.hubness(chassis.corpus_embeddings, bank_dev, k=qb_cfg["k"])
        sims = sims - qb_cfg["beta"] * hub_dev[None, :]
    dense = chassis.topk_from_matrix(dev_qids, sims, len(chassis.doc_ids))
    qrels = {q: ds.train_qrels[q] for q in dev_qids}
    gold_rank = rank_of_gold(qrels, dense)
    extend_k = len(chassis.doc_ids) if args.full_corpus else args.extend_k

    print("[M8] building signature gate over the corpus...", flush=True)
    gate = SignatureGate(raw_corpus)
    print(f"[M8] gate coverage: {gate.coverage()}  eligible={len(gate.eligible)}/{len(raw_corpus)}", flush=True)

    if args.tau is not None:
        # ---- reproduction: apply the recorded dev fit, run nothing on dev ---------------------------------
        best_tau = float(args.tau)
        recorded_path = DOCS_DIR / f"dev_{label}.json"
        recorded = json.loads(recorded_path.read_text()) if recorded_path.exists() else {}
        dev_summary = recorded.get("dev_r3") if recorded.get("best_tau") == best_tau else None
        report = {"rung": rung_label}
        print(f"[M8] tau pinned at {best_tau} (--tau): dev extension verification + sweep skipped"
              + (f"; matches the recorded dev fit in docs/dev_{label}.json" if dev_summary
                 else f"; no recorded dev fit with tau={best_tau} in docs/dev_{label}.json"), flush=True)
    else:
        # ---- verify base + extension once per dev query (cached), then simulate tau -------------------------
        sandbox = VerificationSandbox(default_timeout=args.timeout, reduced_timeout=0.3, workers=args.workers)
        verifier = GatedVerifier(base_k=args.base_k, extend_k=extend_k, tau=1.0, gate=gate, sandbox=sandbox, cache_path=cache_path)
        print(f"[M8] verifying base top-{args.base_k} + extension ranks {args.base_k}..{extend_k} (gate-filtered) for {len(dev_qids)} dev queries...", flush=True)
        t0 = time.time()
        all_qv: Dict[str, QueryVerification] = {}
        base_ids: Dict[str, List[str]] = {}
        ext_ids: Dict[str, List[str]] = {}
        margins: Dict[str, float] = {}
        n_extra = 0
        for i, q in enumerate(dev_qids):
            ordered = sorted(dense[q].items(), key=lambda kv: kv[1], reverse=True)
            margins[q] = UncertaintyRouter.margin(dense[q])
            ids = verifier.candidate_ids(ds.queries_dict[q], ordered)
            base_ids[q], ext_ids[q] = ids[: args.base_k], ids[args.base_k:]
            n_extra += len(ext_ids[q])
            all_qv[q] = verifier.verify_query(ds.queries_dict[q], ids, raw_corpus)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(dev_qids)}  extension candidates so far {n_extra}  ({time.time() - t0:.0f}s)", flush=True)
        verifier.flush_cache()
        ext_s = time.time() - t0
        sandbox.close()

        base_metrics = compute_retrieval_metrics(qrels, {q: TopKVerifier.blend(dense[q], restrict(all_qv[q], base_ids[q]), alpha, len(dense[q])) for q in dev_qids})
        sweep = {}
        for tau in TAU_GRID:
            rr = {}
            for q in dev_qids:
                ids = base_ids[q] + (ext_ids[q] if margins[q] < tau else [])
                rr[q] = TopKVerifier.blend(dense[q], restrict(all_qv[q], ids), alpha, len(dense[q]))
            m = compute_retrieval_metrics(qrels, rr)
            n_ext = sum(1 for q in dev_qids if margins[q] < tau)
            sweep[tau] = {**m, "queries_extended": n_ext}
            print(f"  tau={tau:<6} extended={n_ext:3d}  dev NDCG@10={m['ndcg_at_10']:.4f} (base {base_metrics['ndcg_at_10']:.4f})", flush=True)
        best_tau = max(TAU_GRID, key=lambda t: (sweep[t]["ndcg_at_10"], -sweep[t]["queries_extended"]))
        dev_summary = sweep[best_tau]
        gain = sweep[best_tau]["ndcg_at_10"] - base_metrics["ndcg_at_10"]
        decision = "ADOPT" if gain > 0 and best_tau > 0 else "REJECT"
        report = {
            "rung": rung_label, "model": args.model, "alpha": alpha, "qbnorm": qb_cfg,
            "base_k": args.base_k, "extend_k": extend_k, "gate_coverage": gate.coverage(), "gate_eligible": len(gate.eligible),
            "gold_beyond_base_k": sum(1 for q in dev_qids if gold_rank[q] == 0 or gold_rank[q] > args.base_k),
            "gold_within_extend_k": sum(1 for q in dev_qids if args.base_k < gold_rank[q] <= extend_k),
            "verification_wall_s": round(ext_s, 1), "extension_candidates_total": n_extra,
            "dev_base": base_metrics, "tau_grid": {str(t): sweep[t] for t in TAU_GRID}, "best_tau": best_tau,
            "dev_r3": sweep[best_tau], "gain_ndcg10": round(gain, 5), "decision": decision,
        }
        (DOCS_DIR / f"dev_{label}.json").write_text(json.dumps(report, indent=2))
        print(f"[M8] {decision}: best tau={best_tau} dev NDCG@10 {sweep[best_tau]['ndcg_at_10']:.4f} vs base {base_metrics['ndcg_at_10']:.4f} "
              f"(gold beyond top-{args.base_k}: {report['gold_beyond_base_k']}, within extend: {report['gold_within_extend_k']}) -> docs/dev_{label}.json")
        if args.dev_only or (decision == "REJECT" and not args.force_test):
            return

    # ---- test through mteb ------------------------------------------------------------------
    import mteb

    qb_hook = None
    if args.qbnorm:
        hub_all = QBNorm.hubness(chassis.corpus_embeddings, embs, k=qb_cfg["k"])   # bank = all 5,000 train statements
        qb_hook = QBNorm(hub=hub_all, beta=qb_cfg["beta"])
    sandbox = VerificationSandbox(default_timeout=args.timeout, reduced_timeout=0.3, workers=args.workers)
    verifier = GatedVerifier(base_k=args.base_k, extend_k=extend_k, tau=best_tau, gate=gate, sandbox=sandbox, default_alpha=alpha,
                             cache_path=cache_path)
    name = f"vera/VERA-{'R4-Final-GatedQBNorm' if args.qbnorm else 'R3-Gated'}-{short}"
    model = VERASearchProtocol(chassis=chassis, verifier=verifier, verify_top_k=extend_k, alpha=alpha, qbnorm=qb_hook, name=name)
    t0 = time.time()
    result = mteb.evaluate(model, mteb.get_task("AppsRetrieval"), cache=None, overwrite_strategy="always", co2_tracker=False)
    out = ARTIFACTS_DIR / f"m8_{label}_results.json"
    extra = {"rung": report["rung"], "model": args.model, "alpha": alpha, "top_k": args.base_k, "extend_k": extend_k, "tau": best_tau,
             "beta": qb_cfg["beta"] if qb_cfg else None, "qb_k": qb_cfg["k"] if qb_cfg else None, "router_stats": verifier.stats,
             "wall_time_s": round(time.time() - t0, 1), "timings": model.timings, "dev": dev_summary}
    save_mteb_task_result(result.task_results[0], out, extra=extra)
    data = json.loads(out.read_text())
    ok, errors = validate_mteb_result_schema(data)
    assert ok, errors
    row = data["scores"]["test"][0]
    print(f"\n TEST {report['rung']}  NDCG@10={row['ndcg_at_10']:.4f}  MRR@10={row['mrr_at_10']:.4f}  R@10={row['recall_at_10']:.4f}  router {verifier.stats}")
    if args.promote:
        save_mteb_task_result(data, OFFICIAL_SUBMISSION_JSON)
    verifier.close()


if __name__ == "__main__":
    main()
