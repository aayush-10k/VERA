"""
VERA Milestone 2 Harness Proof-of-Life Test
============================================
Runs end-to-end evaluation using VERASearchProtocol to prove submission mechanics:
1. Indexes APPS corpus and searches test queries.
2. Emits a fully valid MTEB TaskResult JSON.
3. Validates output against MTEB benchmark schema.
4. De-risks submission mechanics before complex components exist.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vera.data.loader import AppsRetrievalDataset
from vera.mtebio.search_protocol import VERASearchProtocol
from vera.mtebio.serializer import save_mteb_task_result, validate_mteb_result_schema

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PROOF_OF_LIFE_JSON = ARTIFACTS_DIR / "m2_proof_of_life.json"
OFFICIAL_SUBMISSION_JSON = PROJECT_ROOT / "appsretrieval_results.json"


def compute_retrieval_metrics(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int] = [1, 3, 5, 10, 20, 100, 1000],
) -> Dict[str, float]:
    """
    Computes standard TREC evaluation metrics (NDCG, MRR, MAP, Recall)
    matching the exact formulas used by MTEB / pytrec_eval.
    """
    ndcg: Dict[int, List[float]] = {k: [] for k in k_values}
    mrr: Dict[int, List[float]] = {k: [] for k in k_values}
    recall: Dict[int, List[float]] = {k: [] for k in k_values}
    map_scores: Dict[int, List[float]] = {k: [] for k in k_values}

    for qid, gold_dict in qrels.items():
        if qid not in results:
            for k in k_values:
                ndcg[k].append(0.0)
                mrr[k].append(0.0)
                recall[k].append(0.0)
                map_scores[k].append(0.0)
            continue

        # Sort candidate doc_ids by predicted score descending
        sorted_preds = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)
        retrieved_ids = [doc_id for doc_id, _ in sorted_preds]

        gold_ids = {doc_id for doc_id, rel in gold_dict.items() if rel > 0}
        total_relevant = len(gold_ids)

        if total_relevant == 0:
            continue

        # Compute metrics for each k
        for k in k_values:
            top_k_ids = retrieved_ids[:k]
            
            # Binary relevance at each rank
            hits = [1 if doc_id in gold_ids else 0 for doc_id in top_k_ids]
            num_hits = sum(hits)

            # Recall@k
            recall[k].append(num_hits / total_relevant)

            # MRR@k (Reciprocal Rank of first relevant hit)
            first_hit_rank = None
            for idx, h in enumerate(hits):
                if h == 1:
                    first_hit_rank = idx + 1
                    break
            mrr[k].append(1.0 / first_hit_rank if first_hit_rank is not None else 0.0)

            # MAP@k
            if num_hits == 0:
                map_scores[k].append(0.0)
            else:
                precisions = []
                running_hits = 0
                for idx, h in enumerate(hits):
                    if h == 1:
                        running_hits += 1
                        precisions.append(running_hits / (idx + 1))
                map_scores[k].append(sum(precisions) / min(k, total_relevant))

            # NDCG@k
            dcg = 0.0
            for idx, h in enumerate(hits):
                if h == 1:
                    dcg += 1.0 / math.log2(idx + 2)

            idcg = 0.0
            for idx in range(min(k, total_relevant)):
                idcg += 1.0 / math.log2(idx + 2)

            ndcg[k].append(dcg / idcg if idcg > 0 else 0.0)

    # Average over all queries
    metrics: Dict[str, float] = {}
    for k in k_values:
        metrics[f"ndcg_at_{k}"] = round(sum(ndcg[k]) / max(1, len(ndcg[k])), 5)
        metrics[f"mrr_at_{k}"] = round(sum(mrr[k]) / max(1, len(mrr[k])), 5)
        metrics[f"recall_at_{k}"] = round(sum(recall[k]) / max(1, len(recall[k])), 5)
        metrics[f"map_at_{k}"] = round(sum(map_scores[k]) / max(1, len(map_scores[k])), 5)

    return metrics


def run_mteb_harness_test(
    sample_queries_only: bool = False,
    sample_size: int = 200,
) -> Dict[str, Any]:
    """
    Executes end-to-end evaluation using VERASearchProtocol.
    Returns a schema-compliant TaskResult dictionary.
    """
    print("[M2 Harness] Loading AppsRetrieval dataset...")
    ds = AppsRetrievalDataset()
    corpus = ds.get_corpus()
    test_queries = ds.get_test_queries()
    test_qrels = ds.get_test_qrels()

    if sample_queries_only:
        print(f"[M2 Harness] Running proof-of-life on sample of {sample_size} queries...")
        sample_qids = list(test_queries.keys())[:sample_size]
        eval_queries = {qid: test_queries[qid] for qid in sample_qids}
        eval_qrels = {qid: test_qrels[qid] for qid in sample_qids if qid in test_qrels}
    else:
        print(f"[M2 Harness] Running full test evaluation over all {len(test_queries)} queries...")
        eval_queries = test_queries
        eval_qrels = test_qrels

    # Initialize model implementing SearchProtocol
    model = VERASearchProtocol(name="VERA-ProofOfLife-Stub")

    # Index corpus
    start_time = time.time()
    print(f"[M2 Harness] Calling model.index() with {len(corpus)} documents...")
    model.index(corpus)
    index_time = time.time() - start_time
    print(f"[M2 Harness] Indexing complete in {index_time:.2f}s.")

    # Execute search
    search_start = time.time()
    print(f"[M2 Harness] Calling model.search() on {len(eval_queries)} queries (top_k=100)...")
    search_results = model.search(eval_queries, top_k=100)
    search_time = time.time() - search_start
    print(f"[M2 Harness] Search complete in {search_time:.2f}s ({(search_time/len(eval_queries))*1000:.1f}ms/query).")

    # Compute official metrics
    print("[M2 Harness] Computing MTEB retrieval metrics...")
    eval_metrics = compute_retrieval_metrics(eval_qrels, search_results)
    eval_metrics["evaluation_time"] = round(search_time, 2)
    eval_metrics["total_queries"] = len(eval_queries)

    # Assemble official MTEB TaskResult dictionary structure
    task_result_dict = {
        "dataset_revision": "CoIR-APPS-v1.0",
        "mteb_dataset_name": "AppsRetrieval",
        "mteb_version": "2.0.1",
        "model_name": model.name,
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test": eval_metrics,
    }

    return task_result_dict


def main():
    parser = argparse.ArgumentParser(description="VERA M2 Harness Proof-of-Life Test")
    parser.add_argument("--sample", action="store_true", help="Run on sample of 200 queries for fast verification")
    parser.add_argument("--sample-size", type=int, default=200, help="Sample size if --sample is set")
    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(" VERA M2 Harness Proof-of-Life Verification")
    print("=" * 65)

    result_dict = run_mteb_harness_test(
        sample_queries_only=args.sample,
        sample_size=args.sample_size,
    )

    # Save to artifacts/m2_proof_of_life.json
    save_mteb_task_result(result_dict, PROOF_OF_LIFE_JSON)
    # Also save to root appsretrieval_results.json as expected by submission guidelines
    save_mteb_task_result(result_dict, OFFICIAL_SUBMISSION_JSON)

    # Validate schema
    print("\n[M2 Harness] Validating generated JSON schema...")
    is_valid, errors = validate_mteb_result_schema(result_dict)

    if is_valid:
        print("[M2 Harness] [PASSED] JSON schema validation succeeded!")
        print(f"[M2 Harness] Sample metrics: NDCG@10={result_dict['test']['ndcg_at_10']}, MRR@10={result_dict['test']['mrr_at_10']}")
    else:
        print("[M2 Harness] [FAILED] Schema errors detected:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    print("=" * 65)
    print(f"Task 02 Proof-of-Life Successful: {PROOF_OF_LIFE_JSON}")
    print("=" * 65)


if __name__ == "__main__":
    main()
