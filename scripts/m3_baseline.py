"""
VERA Milestone 3 — R0 dense baseline through the official MTEB harness
=====================================================================
Runs ``mteb.evaluate`` on ``AppsRetrieval`` with ``VERASearchProtocol`` wrapping the
dense chassis only (no verification). MTEB loads the dataset, drives ``index``/``search``
and scores the run itself, so the emitted JSON *is* an MTEB ``TaskResult``.

Outputs
-------
artifacts/m3_r0_results.json          milestone JSON #1 (full 3,765-query test split)
appsretrieval_results.json            root submission file, overwritten only with --promote
docs/dev_r0.json                      dev-split (500 held-out train pairs) metrics for later ablations

``--backend tfidf`` produces the lexical reference row instead (never promoted).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vera.chassis.baseline import DenseChassis
from vera.chassis.preprocess import ChassisPreprocessor
from vera.data.loader import AppsRetrievalDataset
from vera.eval.metrics import compute_retrieval_metrics
from vera.mtebio.search_protocol import VERASearchProtocol
from vera.mtebio.serializer import save_mteb_task_result, validate_mteb_result_schema

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OFFICIAL_SUBMISSION_JSON = PROJECT_ROOT / "appsretrieval_results.json"


def evaluate_with_mteb(model: VERASearchProtocol, out_path: Path, extra: dict) -> dict:
    import mteb

    task = mteb.get_task("AppsRetrieval")
    t0 = time.time()
    result = mteb.evaluate(model, task, cache=None, overwrite_strategy="always", encode_kwargs={"batch_size": 8}, co2_tracker=False)
    task_result = result.task_results[0]
    extra = dict(extra, wall_time_s=round(time.time() - t0, 1), timings=model.timings)
    save_mteb_task_result(task_result, out_path, extra=extra)
    data = json.loads(out_path.read_text())
    ok, errors = validate_mteb_result_schema(data)
    if not ok:
        raise SystemExit(f"TaskResult schema errors: {errors}")
    return data


def dev_metrics(chassis: DenseChassis, ds: AppsRetrievalDataset, pre: ChassisPreprocessor) -> dict:
    """Dev split: the 500 held-out train queries against the full corpus (no verification)."""
    split = ds.get_or_create_dev_split()
    dev_qids = split["dev_query_ids"]
    dev_queries = pre.process_queries({q: ds.queries_dict[q] for q in dev_qids})
    dev_qrels = {q: ds.train_qrels[q] for q in dev_qids}
    results = chassis.search(dev_queries, top_k=1000, tag="dev_queries")
    return compute_retrieval_metrics(dev_qrels, results)


def main():
    ap = argparse.ArgumentParser(description="VERA R0 baseline via mteb.evaluate")
    ap.add_argument("--model", default="Alibaba-NLP/gte-modernbert-base")
    ap.add_argument("--backend", choices=["st", "tfidf"], default="st")
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--promote", action="store_true", help="also overwrite appsretrieval_results.json")
    ap.add_argument("--no-dev", action="store_true")
    ap.add_argument("--raw-corpus", action="store_true", help="ablation R0b: embed the raw corpus (no boilerplate stripping)")
    ap.add_argument("--no-query-examples", action="store_true", help="ablation: drop the worked-example section from the embedded query")
    args = ap.parse_args()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    chassis = DenseChassis(model_name=args.model, backend=args.backend, max_seq_length=args.max_seq_length, batch_size=args.batch_size)
    short = args.model.split("/")[-1]
    name = f"vera/VERA-R0-{short}" if args.backend == "st" else "vera/VERA-REF-tfidf"
    model = VERASearchProtocol(chassis=chassis, name=name, remove_examples_from_query=args.no_query_examples)
    if args.raw_corpus:
        model.preprocessor.process_corpus = lambda corpus: {**corpus}  # type: ignore[method-assign]
        name += "-rawcorpus"
        model.name = name

    suffix = ("_raw" if args.raw_corpus else "") + ("_noqex" if args.no_query_examples else "")
    out = ARTIFACTS_DIR / (f"m3_r0{suffix}_results.json" if args.backend == "st" else "ref_tfidf_results.json")
    print("=" * 66)
    print(f" VERA M3 — R0 baseline ({name}) via mteb.evaluate")
    print("=" * 66)
    data = evaluate_with_mteb(model, out, extra={"rung": ("R0b" if suffix else "R0") if args.backend == "st" else "REF", "backend": args.backend,
                                                  "model": args.model, "max_seq_length": args.max_seq_length,
                                                  "corpus_preprocessing": "raw" if args.raw_corpus else "strip_corpus_boilerplate",
                                                  "query_examples_embedded": not args.no_query_examples, "verification": None})
    row = data["scores"]["test"][0]
    print(f"\n TEST  NDCG@10={row['ndcg_at_10']:.4f}  MRR@10={row['mrr_at_10']:.4f}  R@10={row['recall_at_10']:.4f}  R@100={row['recall_at_100']:.4f}  R@150≈R@100..1000")

    if not args.no_dev:
        ds = AppsRetrievalDataset()
        pre = ChassisPreprocessor()
        # chassis is already indexed by mteb's index() call (same preprocessing) -> reuse
        dm = dev_metrics(chassis, ds, pre)
        dev_path = PROJECT_ROOT / "docs" / ("dev_r0.json" if args.backend == "st" else "dev_ref_tfidf.json")
        dev_path.write_text(json.dumps({"rung": data["vera_run"]["rung"], "model": args.model, "dev": dm}, indent=2))
        print(f" DEV   NDCG@10={dm['ndcg_at_10']:.4f}  MRR@10={dm['mrr_at_10']:.4f}  R@100={dm['recall_at_100']:.4f}  -> {dev_path}")

    if args.promote and args.backend == "st" and not suffix:
        save_mteb_task_result(data, OFFICIAL_SUBMISSION_JSON)
    print(f" artifact: {out}")


if __name__ == "__main__":
    main()
