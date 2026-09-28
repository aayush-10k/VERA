"""
VERA Milestone 8 — R3: gated corpus-wide verification (conditional on the M4 gate)
=================================================================================
The M4 gate passed (docs/goldrun.md: 86.7% projected), so this rung is evaluated.

Design (Plan.md 7.4): an **uncertainty router** skips the extension when the dense
top-1/top-2 margin is decisive; otherwise verification extends from the dense top-150 to
dense ranks 150..``extend_k`` (``--full-corpus`` = every gate-eligible document) filtered by
the **static signature gate** (program I/O shape must be compatible with the example's
input layout). A runtime budget projection is printed before anything runs.

Dev protocol: reuse the cached R2 verification of the dense top-150 (from
``scripts/m7_topk_verify.py``), verify the extension for the uncertain dev queries, sweep
``tau`` and report dev NDCG@10 of R3 against R2. If R3 does not beat R2 on dev, the
rejection is recorded (``docs/dev_r3.json``) and no test JSON is produced unless ``--force-test``.
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
from vera.data.loader import AppsRetrievalDataset
from vera.eval.metrics import compute_retrieval_metrics, rank_of_gold
from vera.gate.router import SignatureGate, UncertaintyRouter
from vera.mtebio.search_protocol import VERASearchProtocol
from vera.mtebio.serializer import save_mteb_task_result, validate_mteb_result_schema
from vera.verify.boost import GatedVerifier, QueryVerification, TopKVerifier, compute_rarity_confidence
from vera.verify.executor import VerificationSandbox

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DOCS_DIR = PROJECT_ROOT / "docs"
R3_RESULTS_JSON = ARTIFACTS_DIR / "m8_r3_results.json"
OFFICIAL_SUBMISSION_JSON = PROJECT_ROOT / "appsretrieval_results.json"
TAU_GRID = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 1.0]  # 1.0 == extend every query


def load_dev_verification(path: Path) -> Dict[str, QueryVerification]:
    raw = json.loads(path.read_text())
    out = {}
    for q, d in raw.items():
        qv = QueryVerification(n_examples=d["n_examples"])
        qv.conf, qv.passed, qv.m_first, qv.n_all_pass = d["conf"], d["passed"], d["m_first"], d["n_all_pass"]
        out[q] = qv
    return out


def main():
    ap = argparse.ArgumentParser(description="VERA R3 gated corpus-wide verification")
    ap.add_argument("--model", default="Alibaba-NLP/gte-modernbert-base")
    ap.add_argument("--base-k", type=int, default=150)
    ap.add_argument("--extend-k", type=int, default=1000)
    ap.add_argument("--full-corpus", action="store_true")
    ap.add_argument("--alpha", type=float, default=None, help="alpha (default: selected by m7 in docs/dev_r2.json)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=0.75)
    ap.add_argument("--force-test", action="store_true")
    ap.add_argument("--dev-only", action="store_true")
    ap.add_argument("--promote", action="store_true")
    args = ap.parse_args()

    dev_r2 = json.loads((DOCS_DIR / "dev_r2.json").read_text())
    alpha = args.alpha if args.alpha is not None else float(dev_r2["selected_alpha"])

    ds = AppsRetrievalDataset()
    pre = ChassisPreprocessor()
    raw_corpus = ds.get_corpus()
    chassis = DenseChassis(model_name=args.model, batch_size=8)
    chassis.index_corpus(pre.process_corpus(raw_corpus))
    verif_cache = chassis.cache_dir / f"dev_r2_verification_{args.model.split('/')[-1]}_k{args.base_k}.json"
    base_qvs = load_dev_verification(verif_cache)

    # dev dense results (full ranking) from the cached train-query embeddings
    split = ds.get_or_create_dev_split()
    dev_qids = split["dev_query_ids"]
    all_train = pre.process_queries({q: ds.queries_dict[q] for q in sorted(ds.train_qrels)})
    qids, embs = chassis.encode_queries(all_train, tag="train_queries")
    pos = {q: i for i, q in enumerate(qids)}
    sims = chassis.similarity(embs[[pos[q] for q in dev_qids]])
    dense = chassis.topk_from_matrix(dev_qids, sims, len(chassis.doc_ids))
    qrels = {q: ds.train_qrels[q] for q in dev_qids}
    gold_rank = rank_of_gold(qrels, dense)

    print("[M8] building signature gate over the corpus...", flush=True)
    gate = SignatureGate(raw_corpus)
    print(f"[M8] gate coverage: {gate.coverage()}  eligible={len(gate.eligible)}/{len(raw_corpus)}", flush=True)

    # ---- budget projection + extension verification for uncertain queries ------------------
    margins = {q: UncertaintyRouter.margin(dense[q]) for q in dev_qids}
    extend_k = len(chassis.doc_ids) if args.full_corpus else args.extend_k
    sandbox = VerificationSandbox(default_timeout=args.timeout, reduced_timeout=0.3, workers=args.workers)
    verifier = GatedVerifier(base_k=args.base_k, extend_k=extend_k, tau=1.0, gate=gate, sandbox=sandbox)

    # verify the extension once for every dev query (tau=1.0), then simulate each tau by masking
    print(f"[M8] verifying extension ranks {args.base_k}..{extend_k} (gate-filtered) for {len(dev_qids)} dev queries...", flush=True)
    t0 = time.time()
    ext_qvs: Dict[str, QueryVerification] = {}
    n_extra = 0
    for i, q in enumerate(dev_qids):
        ordered = sorted(dense[q].items(), key=lambda kv: kv[1], reverse=True)
        ids = verifier.candidate_ids(ds.queries_dict[q], ordered)[args.base_k:]
        n_extra += len(ids)
        ext_qvs[q] = verifier.verify_query(ds.queries_dict[q], ids, raw_corpus) if ids else QueryVerification(n_examples=0)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(dev_qids)}  extra candidates so far {n_extra}  ({time.time() - t0:.0f}s)", flush=True)
    ext_s = time.time() - t0
    sandbox.close()

    # ---- merge base + extension confidences, recompute rarity with the combined m ----------
    def merged(q: str, extended: bool) -> QueryVerification:
        b = base_qvs[q]
        if not extended or ext_qvs[q].n_examples == 0:
            return b
        e = ext_qvs[q]
        qv = QueryVerification(n_examples=b.n_examples)
        m = b.m_first + e.m_first
        for src in (b, e):
            for d, p in src.passed.items():
                qv.passed[d] = p
                qv.conf[d] = compute_rarity_confidence(p, b.n_examples, m if p > 0 else 1)
        qv.m_first, qv.n_all_pass = m, b.n_all_pass + e.n_all_pass
        return qv

    r2 = {q: TopKVerifier.blend(dense[q], base_qvs[q], alpha, args.base_k) for q in dev_qids}
    r2_metrics = compute_retrieval_metrics(qrels, r2)
    sweep = {}
    for tau in TAU_GRID:
        r3 = {q: TopKVerifier.blend(dense[q], merged(q, margins[q] < tau), alpha, len(dense[q])) for q in dev_qids}
        m = compute_retrieval_metrics(qrels, r3)
        n_ext = sum(1 for q in dev_qids if margins[q] < tau)
        sweep[tau] = {**m, "queries_extended": n_ext}
        print(f"  tau={tau:<6} extended={n_ext:3d}  dev NDCG@10={m['ndcg_at_10']:.4f} (R2 {r2_metrics['ndcg_at_10']:.4f})", flush=True)
    best_tau = max(TAU_GRID, key=lambda t: (sweep[t]["ndcg_at_10"], -sweep[t]["queries_extended"]))
    gain = sweep[best_tau]["ndcg_at_10"] - r2_metrics["ndcg_at_10"]
    decision = "ADOPT" if gain > 0 else "REJECT"
    report = {
        "rung": "R3", "model": args.model, "alpha": alpha, "base_k": args.base_k, "extend_k": extend_k,
        "gate_coverage": gate.coverage(), "gate_eligible": len(gate.eligible),
        "gold_beyond_base_k": sum(1 for q in dev_qids if gold_rank[q] == 0 or gold_rank[q] > args.base_k),
        "gold_within_extend_k": sum(1 for q in dev_qids if args.base_k < gold_rank[q] <= extend_k),
        "extension_wall_s": round(ext_s, 1), "extra_candidates_total": n_extra,
        "dev_r2": r2_metrics, "tau_grid": {str(t): sweep[t] for t in TAU_GRID}, "best_tau": best_tau,
        "dev_r3": sweep[best_tau], "gain_ndcg10": round(gain, 5), "decision": decision,
    }
    (DOCS_DIR / "dev_r3.json").write_text(json.dumps(report, indent=2))
    print(f"[M8] {decision}: best tau={best_tau} dev NDCG@10 {sweep[best_tau]['ndcg_at_10']:.4f} vs R2 {r2_metrics['ndcg_at_10']:.4f} "
          f"(gold beyond top-{args.base_k}: {report['gold_beyond_base_k']}, within extend: {report['gold_within_extend_k']}) -> docs/dev_r3.json")
    if args.dev_only or (decision == "REJECT" and not args.force_test):
        return

    # ---- test through mteb ------------------------------------------------------------------
    import mteb

    sandbox = VerificationSandbox(default_timeout=args.timeout, reduced_timeout=0.3, workers=args.workers)
    verifier = GatedVerifier(base_k=args.base_k, extend_k=extend_k, tau=best_tau, gate=gate, sandbox=sandbox, default_alpha=alpha,
                             cache_path=str(chassis.cache_dir / f"r3_verification_{args.model.split('/')[-1]}_k{args.base_k}_e{extend_k}.json"))
    model = VERASearchProtocol(chassis=chassis, verifier=verifier, verify_top_k=extend_k, alpha=alpha,
                               name=f"vera/VERA-R3-GatedCorpusWide-{args.model.split('/')[-1]}")
    t0 = time.time()
    result = mteb.evaluate(model, mteb.get_task("AppsRetrieval"), cache=None, overwrite_strategy="always", co2_tracker=False)
    extra = {"rung": "R3", "model": args.model, "alpha": alpha, "top_k": args.base_k, "extend_k": extend_k, "tau": best_tau,
             "router_stats": verifier.stats, "wall_time_s": round(time.time() - t0, 1), "timings": model.timings, "dev": sweep[best_tau]}
    save_mteb_task_result(result.task_results[0], R3_RESULTS_JSON, extra=extra)
    data = json.loads(R3_RESULTS_JSON.read_text())
    ok, errors = validate_mteb_result_schema(data)
    assert ok, errors
    row = data["scores"]["test"][0]
    print(f"\n TEST R3  NDCG@10={row['ndcg_at_10']:.4f}  MRR@10={row['mrr_at_10']:.4f}  extended {verifier.stats}")
    if args.promote:
        save_mteb_task_result(data, OFFICIAL_SUBMISSION_JSON)
    sandbox.close()


if __name__ == "__main__":
    main()
