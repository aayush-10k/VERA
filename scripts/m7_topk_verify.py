"""
VERA Milestone 7 — R2: top-K verification re-ranking with the calibrated rarity boost
====================================================================================
1. Dense candidates (top-150) for the 500 dev queries from the cached chassis embeddings.
2. Verify every dev candidate once; grid-search alpha in [0.05, 0.40] (+ a wider tail)
   on dev NDCG@10; record the dev ablation (R0 vs R2 at every alpha).
3. Run the test split through ``mteb.evaluate`` with the selected alpha and emit
   ``artifacts/m7_r2_results.json`` (milestone JSON #3); ``--promote`` copies it to
   ``appsretrieval_results.json``.

Only dev is used for fitting. The test split is touched once, at the end.
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
from vera.mtebio.search_protocol import VERASearchProtocol
from vera.mtebio.serializer import save_mteb_task_result, validate_mteb_result_schema
from vera.verify.boost import TopKVerifier

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DOCS_DIR = PROJECT_ROOT / "docs"
R2_RESULTS_JSON = ARTIFACTS_DIR / "m7_r2_results.json"
OFFICIAL_SUBMISSION_JSON = PROJECT_ROOT / "appsretrieval_results.json"
DEV_R2_JSON = DOCS_DIR / "dev_r2.json"

ALPHA_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.75, 1.0]


def dev_candidates(chassis: DenseChassis, ds: AppsRetrievalDataset, pre: ChassisPreprocessor, top_k: int):
    """Dense results for the 500 dev queries, sliced out of the cached 5,000 train-query embeddings."""
    split = ds.get_or_create_dev_split()
    dev_qids = split["dev_query_ids"]
    all_train = pre.process_queries({q: ds.queries_dict[q] for q in sorted(ds.train_qrels)})
    qids, embs = chassis.encode_queries(all_train, tag="train_queries")
    pos = {q: i for i, q in enumerate(qids)}
    dev_embs = embs[[pos[q] for q in dev_qids]]
    sims = chassis.similarity(dev_embs)
    dense = chassis.topk_from_matrix(dev_qids, sims, top_k)
    qrels = {q: ds.train_qrels[q] for q in dev_qids}
    return dev_qids, dense, qrels


def main():
    ap = argparse.ArgumentParser(description="VERA R2 top-K verification")
    ap.add_argument("--model", default="Alibaba-NLP/gte-modernbert-base")
    ap.add_argument("--top-k", type=int, default=150)
    ap.add_argument("--alpha", type=float, default=None, help="fixed alpha (skips the dev grid search)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=0.75)
    ap.add_argument("--dev-only", action="store_true")
    ap.add_argument("--promote", action="store_true")
    args = ap.parse_args()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    ds = AppsRetrievalDataset()
    pre = ChassisPreprocessor()
    raw_corpus = ds.get_corpus()
    chassis = DenseChassis(model_name=args.model, batch_size=8)
    chassis.index_corpus(pre.process_corpus(raw_corpus))

    from vera.verify.executor import VerificationSandbox

    verifier = TopKVerifier(sandbox=VerificationSandbox(default_timeout=args.timeout, reduced_timeout=0.3, workers=args.workers))

    # ---- dev: verify once, sweep alpha -------------------------------------------
    selected_alpha = args.alpha
    dev_report: Dict[str, object] = {}
    if selected_alpha is None or args.dev_only:
        dev_qids, dev_dense, dev_qrels = dev_candidates(chassis, ds, pre, top_k=max(args.top_k, 1000))
        print(f"[M7] verifying dense top-{args.top_k} for {len(dev_qids)} dev queries...", flush=True)
        t0 = time.time()
        qvs = {}
        for i, qid in enumerate(dev_qids):
            ordered = sorted(dev_dense[qid].items(), key=lambda kv: kv[1], reverse=True)[: args.top_k]
            qvs[qid] = verifier.verify_query(ds.queries_dict[qid], [d for d, _ in ordered], raw_corpus)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(dev_qids)} ({time.time() - t0:.0f}s)", flush=True)
        verify_s = time.time() - t0
        # cache the alpha-independent verification so R3/R4 scripts can reuse it without re-running code
        verif_cache = chassis.cache_dir / f"dev_r2_verification_{args.model.split('/')[-1]}_k{args.top_k}.json"
        verif_cache.write_text(json.dumps({q: {"n_examples": qv.n_examples, "conf": qv.conf, "passed": qv.passed,
                                               "m_first": qv.m_first, "n_all_pass": qv.n_all_pass} for q, qv in qvs.items()}))
        print(f"[M7] cached dev verification -> {verif_cache}", flush=True)

        base_metrics = compute_retrieval_metrics(dev_qrels, dev_dense)
        gold_rank_dense = rank_of_gold(dev_qrels, dev_dense)
        sweep = {}
        for alpha in ALPHA_GRID:
            reranked = {q: TopKVerifier.blend(dev_dense[q], qvs[q], alpha, args.top_k) for q in dev_qids}
            m = compute_retrieval_metrics(dev_qrels, reranked)
            sweep[alpha] = m
            print(f"  alpha={alpha:.2f}  dev NDCG@10={m['ndcg_at_10']:.4f}  MRR@10={m['mrr_at_10']:.4f}  R@10={m['recall_at_10']:.4f}", flush=True)
        best_alpha = max(ALPHA_GRID, key=lambda a: (sweep[a]["ndcg_at_10"], -a))
        if selected_alpha is None:
            selected_alpha = best_alpha

        # diagnostics: how often does the gold sit in the verified top-K, does it pass, who else passes
        n_gold_in_topk = sum(1 for q in dev_qids if 0 < gold_rank_dense[q] <= args.top_k)
        n_gold_pass = sum(1 for q in dev_qids if qvs[q].passed.get(next(iter(dev_qrels[q])), 0) == qvs[q].n_examples > 0)
        n_with_examples = sum(1 for q in dev_qids if qvs[q].n_examples > 0)
        n_unique_pass = sum(1 for q in dev_qids if qvs[q].n_all_pass == 1)
        dev_report = {
            "rung": "R2", "model": args.model, "top_k": args.top_k, "timeout_s": args.timeout,
            "verify_wall_s": round(verify_s, 1), "alpha_grid": {str(a): sweep[a] for a in ALPHA_GRID},
            "dev_r0": base_metrics, "best_alpha": best_alpha, "selected_alpha": selected_alpha,
            "dev_r2": sweep[selected_alpha],
            "diagnostics": {
                "dev_queries": len(dev_qids), "queries_with_examples": n_with_examples,
                "gold_in_dense_topk": n_gold_in_topk, "gold_passes_own_examples": n_gold_pass,
                "queries_with_exactly_one_full_passer": n_unique_pass,
                "mean_candidates_passing_all": round(sum(qv.n_all_pass for qv in qvs.values()) / len(qvs), 2),
            },
        }
        DEV_R2_JSON.write_text(json.dumps(dev_report, indent=2))
        print(f"[M7] dev R0 NDCG@10={base_metrics['ndcg_at_10']:.4f} -> R2 NDCG@10={sweep[selected_alpha]['ndcg_at_10']:.4f} "
              f"at alpha={selected_alpha} (best {best_alpha}); gold in top-{args.top_k}: {n_gold_in_topk}/{len(dev_qids)}, "
              f"gold passes: {n_gold_pass}; -> {DEV_R2_JSON}")
        if args.dev_only:
            verifier.close()
            return

    # ---- test through mteb ------------------------------------------------------------
    import mteb

    verifier.alpha = selected_alpha
    model = VERASearchProtocol(chassis=chassis, verifier=verifier, verify_top_k=args.top_k, alpha=selected_alpha,
                               name=f"vera/VERA-R2-TopKVerify-{args.model.split('/')[-1]}")
    task = mteb.get_task("AppsRetrieval")
    t0 = time.time()
    result = mteb.evaluate(model, task, cache=None, overwrite_strategy="always", co2_tracker=False, encode_kwargs={"batch_size": 8})
    tr = result.task_results[0]
    extra = {"rung": "R2", "model": args.model, "top_k": args.top_k, "alpha": selected_alpha, "timeout_s": args.timeout,
             "wall_time_s": round(time.time() - t0, 1), "timings": model.timings, "dev": dev_report.get("dev_r2")}
    save_mteb_task_result(tr, R2_RESULTS_JSON, extra=extra)
    data = json.loads(R2_RESULTS_JSON.read_text())
    ok, errors = validate_mteb_result_schema(data)
    assert ok, errors
    row = data["scores"]["test"][0]
    print(f"\n TEST R2  NDCG@10={row['ndcg_at_10']:.4f}  MRR@10={row['mrr_at_10']:.4f}  R@10={row['recall_at_10']:.4f}  alpha={selected_alpha}")
    if args.promote:
        save_mteb_task_result(data, OFFICIAL_SUBMISSION_JSON)
    verifier.close()


if __name__ == "__main__":
    main()
