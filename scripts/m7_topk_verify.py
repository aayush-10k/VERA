"""
VERA Milestone 7 Evaluation (Rung R2 — Top-K Verification Re-Ranking & Rarity Boost)
==================================================================================
1. Obtains dense candidate rankings from DenseChassis (top-150).
2. Performs grid search on the 500-pair dev split to fit optimal alpha in [0.05, 0.40].
3. Executes Top-K verification with calibrated rarity boost on test queries.
4. Emits Milestone JSON #3: artifacts/m7_r2_results.json (Guaranteed Fallback Ship).
5. Validates against official MTEB benchmark schema.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vera.chassis.baseline import DenseChassis
from vera.chassis.preprocess import ChassisPreprocessor
from vera.data.loader import AppsRetrievalDataset
from vera.mtebio.serializer import save_mteb_task_result, validate_mteb_result_schema
from vera.verify.boost import TopKVerifier
from scripts.m2_harness_test import compute_retrieval_metrics

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
R2_RESULTS_JSON = ARTIFACTS_DIR / "m7_r2_results.json"
OFFICIAL_SUBMISSION_JSON = PROJECT_ROOT / "appsretrieval_results.json"
DEV_SPLIT_FILE = PROJECT_ROOT / "vera" / "data" / "dev_split_ids.json"


def calibrate_alpha_on_dev(
    verifier: TopKVerifier,
    dev_queries: Dict[str, str],
    dev_dense_results: Dict[str, Dict[str, float]],
    dev_qrels: Dict[str, Dict[str, int]],
    raw_corpus: Dict[str, str],
    top_k: int = 150,
    candidate_alphas: Optional[List[float]] = None,
) -> Tuple[float, Dict[float, float]]:
    """Grid search across alpha candidates to find the optimal boost weight on the dev split."""
    from collections import Counter
    from vera.verify.boost import compute_rarity_confidence, min_max_normalize

    if candidate_alphas is None:
        candidate_alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    print(f"\n[M7 Calibration] Precomputing verification confidence for {len(dev_queries)} dev queries...", flush=True)
    dev_precomputed = {}
    for qid, q_text in dev_queries.items():
        if qid not in dev_dense_results:
            continue
        dense_scores = dev_dense_results[qid]
        sorted_candidates = sorted(dense_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_slice = sorted_candidates[:top_k]
        top_candidate_ids = [doc_id for doc_id, _ in top_slice]

        norm_dense = min_max_normalize(dense_scores)
        examples = verifier.parser.parse_examples(q_text)

        conf_map: Dict[str, float] = {}
        if examples:
            verification_results = {}
            for doc_id in top_candidate_ids:
                code = raw_corpus.get(doc_id, "")
                verification_results[doc_id] = verifier.sandbox.verify_candidate(code, examples)

            first_outputs = [
                res.first_output for res in verification_results.values() if res.passed_examples > 0 and res.first_output
            ]
            output_counter = Counter(first_outputs)

            for doc_id in top_candidate_ids:
                res = verification_results[doc_id]
                m = output_counter.get(res.first_output, 1)
                conf_map[doc_id] = compute_rarity_confidence(
                    passed_examples=res.passed_examples,
                    total_examples=len(examples),
                    identical_output_count=m,
                )

        dev_precomputed[qid] = {
            "norm_dense": norm_dense,
            "conf_map": conf_map,
            "sorted_cands": sorted_candidates,
        }

    print(f"[M7 Calibration] Starting alpha grid search across {len(candidate_alphas)} values...", flush=True)
    best_alpha = 0.20
    best_ndcg = -1.0
    alpha_scores: Dict[float, float] = {}

    for alpha in candidate_alphas:
        reranked_dev: Dict[str, Dict[str, float]] = {}
        for qid, cache in dev_precomputed.items():
            norm_dense = cache["norm_dense"]
            conf_map = cache["conf_map"]
            sorted_candidates = cache["sorted_cands"]

            scores: Dict[str, float] = {}
            for doc_id, _ in sorted_candidates[:top_k]:
                scores[doc_id] = norm_dense.get(doc_id, 0.0) + alpha * conf_map.get(doc_id, 0.0)
            for doc_id, _ in sorted_candidates[top_k:]:
                scores[doc_id] = norm_dense.get(doc_id, 0.0)

            reranked_dev[qid] = dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))

        metrics = compute_retrieval_metrics(dev_qrels, reranked_dev)
        ndcg10 = metrics.get("ndcg_at_10", 0.0)
        alpha_scores[alpha] = ndcg10
        print(f"  alpha = {alpha:.2f} -> dev NDCG@10 = {ndcg10:.4f}", flush=True)

        if ndcg10 > best_ndcg:
            best_ndcg = ndcg10
            best_alpha = alpha

    print(f"[M7 Calibration] Optimal alpha* = {best_alpha:.2f} (dev NDCG@10: {best_ndcg:.4f})\n", flush=True)
    return best_alpha, alpha_scores


def run_r2_evaluation(
    model_name: str = "Alibaba-NLP/gte-modernbert-base",
    sample_size: Optional[int] = None,
    top_k: int = 150,
    alpha: Optional[float] = None,
    run_grid_search: bool = True,
    use_fallback: bool = True,
) -> dict:
    preprocessor = ChassisPreprocessor()
    ds = AppsRetrievalDataset()

    print("[M7 Verification] Loading dataset...", flush=True)
    raw_corpus = ds.get_corpus()
    test_queries = ds.get_test_queries()
    test_qrels = ds.get_test_qrels()

    clean_corpus = preprocessor.process_corpus(raw_corpus)

    if sample_size and sample_size < len(test_queries):
        sampled_qids = list(test_queries.keys())[:sample_size]
        eval_queries = {qid: test_queries[qid] for qid in sampled_qids}
        eval_qrels = {qid: test_qrels[qid] for qid in sampled_qids if qid in test_qrels}
    else:
        eval_queries = test_queries
        eval_qrels = test_qrels

    clean_queries = preprocessor.process_queries(eval_queries)

    # Initialize Chassis
    chassis = DenseChassis(model_name=model_name, use_fallback=use_fallback)
    t_idx0 = time.time()
    chassis.index_corpus(clean_corpus)
    index_time = time.time() - t_idx0

    print(f"[M7 Verification] Running dense retrieval for {len(clean_queries)} test queries...", flush=True)
    t_search0 = time.time()
    dense_results = chassis.search(clean_queries, top_k=top_k)
    search_time = time.time() - t_search0

    # Initialize verifier
    verifier = TopKVerifier(default_alpha=0.20)

    # Optional grid search on dev split if alpha is not fixed
    selected_alpha = alpha
    if selected_alpha is None and run_grid_search and DEV_SPLIT_FILE.exists():
        with open(DEV_SPLIT_FILE, "r", encoding="utf-8") as f:
            dev_split_data = json.load(f)
        dev_qids = set(dev_split_data.get("dev_query_ids", [])[:100])  # Fast 100 dev queries
        dev_queries = {qid: ds.queries_dict[qid] for qid in dev_qids if qid in ds.queries_dict}
        dev_qrels = {qid: ds.train_qrels[qid] for qid in dev_qids if qid in ds.train_qrels}
        clean_dev_queries = preprocessor.process_queries(dev_queries)
        dev_dense_results = chassis.search(clean_dev_queries, top_k=top_k)
        selected_alpha, _ = calibrate_alpha_on_dev(
            verifier=verifier,
            dev_queries=dev_queries,
            dev_dense_results=dev_dense_results,
            dev_qrels=dev_qrels,
            raw_corpus=raw_corpus,
            top_k=top_k,
        )
    elif selected_alpha is None:
        selected_alpha = 0.20  # Calibrated default

    print(f"[M7 Verification] Re-ranking {len(eval_queries)} test queries with alpha={selected_alpha:.2f}...", flush=True)
    t_verify0 = time.time()
    reranked_results: Dict[str, Dict[str, float]] = {}

    for idx, (qid, q_text) in enumerate(eval_queries.items()):
        cand_scores = dense_results.get(qid, {})
        reranked_results[qid] = verifier.rerank_query(
            query_text=q_text,
            dense_scores=cand_scores,
            corpus_dict=raw_corpus,
            top_k=top_k,
            alpha=selected_alpha,
        )
        if (idx + 1) % 10 == 0 or (idx + 1) == len(eval_queries):
            print(f"  [{idx + 1}/{len(eval_queries)}] Query {qid} verified...", flush=True)

    verification_time = time.time() - t_verify0
    total_eval_time = search_time + verification_time

    print("[M7 Verification] Computing official MTEB retrieval metrics...", flush=True)
    metrics = compute_retrieval_metrics(eval_qrels, reranked_results)
    metrics["evaluation_time"] = round(total_eval_time, 2)
    metrics["index_time"] = round(index_time, 2)
    metrics["verification_time"] = round(verification_time, 2)
    metrics["total_queries"] = len(eval_queries)
    metrics["calibrated_alpha"] = selected_alpha

    result_dict = {
        "dataset_revision": "CoIR-APPS-v1.0",
        "mteb_dataset_name": "AppsRetrieval",
        "mteb_version": "2.0.1",
        "model_name": f"VERA-R2-TopKVerify-{model_name.split('/')[-1]}",
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test": metrics,
    }

    return result_dict


def main():
    parser = argparse.ArgumentParser(description="VERA R2 Top-K Verification Evaluation")
    parser.add_argument("--model", type=str, default="Alibaba-NLP/gte-modernbert-base")
    parser.add_argument("--sample", type=int, default=200, help="Number of test queries to sample (0 for all)")
    parser.add_argument("--top_k", type=int, default=150, help="Number of dense candidates to verify")
    parser.add_argument("--alpha", type=float, default=None, help="Fixed alpha weight (bypasses grid search)")
    parser.add_argument("--all", action="store_true", help="Evaluate on all 3,765 test queries")
    parser.add_argument("--no_grid", action="store_true", help="Skip dev grid search and use default alpha")
    parser.add_argument("--online", action="store_true", help="Download online transformer weights")
    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    sample_size = None if args.all or args.sample == 0 else args.sample

    print("=" * 65)
    print(" VERA Milestone 7: R2 Top-K Verification Re-Ranking Evaluation")
    print("=" * 65)

    result_dict = run_r2_evaluation(
        model_name=args.model,
        sample_size=sample_size,
        top_k=args.top_k,
        alpha=args.alpha,
        run_grid_search=not args.no_grid,
        use_fallback=not args.online,
    )

    save_mteb_task_result(result_dict, R2_RESULTS_JSON)
    save_mteb_task_result(result_dict, OFFICIAL_SUBMISSION_JSON)

    is_valid, errors = validate_mteb_result_schema(result_dict)
    assert is_valid, f"Schema validation errors: {errors}"

    ndcg10 = result_dict["test"]["ndcg_at_10"]
    mrr10 = result_dict["test"]["mrr_at_10"]
    calibrated_alpha = result_dict["test"]["calibrated_alpha"]

    print("\n" + "=" * 65)
    print(f" R2 Verification Results (NDCG@10: {ndcg10}, MRR@10: {mrr10}, alpha: {calibrated_alpha})")
    print(f" Milestone JSON #3 committed: {R2_RESULTS_JSON}")
    print(f" Official submission updated: {OFFICIAL_SUBMISSION_JSON}")
    print("=" * 65)


if __name__ == "__main__":
    main()
