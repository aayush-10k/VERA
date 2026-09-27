"""
VERA Milestone 9 — R4: QB-Norm demotion + final calibration
==========================================================
Bank = the public train problem statements (content only). For a corpus document d,
``hub(d)`` = mean of its top-k cosine similarities to the bank; the dense score is
demoted by ``beta * hub(d)`` before normalisation and the verification boost.

Dev protocol: the 500 dev statements are removed from the bank (otherwise their own golds
would be demoted, which does not happen for test queries). (k, beta) are swept on dev for
the full pipeline (dense + cached R2 verification of the top-150) and the best pair is kept
only if it beats R2; otherwise the rejection is recorded. Test: bank = all 5,000 train
statements, run through ``mteb.evaluate`` -> ``artifacts/m9_r4_results.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vera.chassis.baseline import DenseChassis
from vera.chassis.preprocess import ChassisPreprocessor
from vera.chassis.qbnorm import QBNorm
from vera.data.loader import AppsRetrievalDataset
from vera.eval.metrics import compute_retrieval_metrics
from vera.mtebio.search_protocol import VERASearchProtocol
from vera.mtebio.serializer import save_mteb_task_result, validate_mteb_result_schema
from vera.verify.boost import TopKVerifier
from vera.verify.executor import VerificationSandbox

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DOCS_DIR = PROJECT_ROOT / "docs"
R4_RESULTS_JSON = ARTIFACTS_DIR / "m9_r4_results.json"
OFFICIAL_SUBMISSION_JSON = PROJECT_ROOT / "appsretrieval_results.json"
K_GRID = [1, 3, 5]
BETA_GRID = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]


def main():
    ap = argparse.ArgumentParser(description="VERA R4 QB-Norm")
    ap.add_argument("--model", default="Alibaba-NLP/gte-modernbert-base")
    ap.add_argument("--top-k", type=int, default=150)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=0.75)
    ap.add_argument("--force-test", action="store_true")
    ap.add_argument("--promote", action="store_true")
    args = ap.parse_args()

    from scripts.m8_corpus_gate import load_dev_verification

    dev_r2 = json.loads((DOCS_DIR / "dev_r2.json").read_text())
    alpha = args.alpha if args.alpha is not None else float(dev_r2["selected_alpha"])

    ds = AppsRetrievalDataset()
    pre = ChassisPreprocessor()
    raw_corpus = ds.get_corpus()
    chassis = DenseChassis(model_name=args.model, batch_size=8)
    chassis.index_corpus(pre.process_corpus(raw_corpus))
    base_qvs = load_dev_verification(chassis.cache_dir / f"dev_r2_verification_{args.model.split('/')[-1]}_k{args.top_k}.json")

    split = ds.get_or_create_dev_split()
    dev_qids = split["dev_query_ids"]
    dev_set = set(dev_qids)
    all_train = pre.process_queries({q: ds.queries_dict[q] for q in sorted(ds.train_qrels)})
    qids, embs = chassis.encode_queries(all_train, tag="train_queries")
    pos = {q: i for i, q in enumerate(qids)}
    dev_embs = embs[[pos[q] for q in dev_qids]]
    bank_dev = embs[[pos[q] for q in qids if q not in dev_set]]      # bank without the dev statements
    sims = chassis.similarity(dev_embs)
    qrels = {q: ds.train_qrels[q] for q in dev_qids}
    n_docs = len(chassis.doc_ids)

    dense_r0 = chassis.topk_from_matrix(dev_qids, sims, n_docs)
    r0 = compute_retrieval_metrics(qrels, dense_r0)
    r2 = compute_retrieval_metrics(qrels, {q: TopKVerifier.blend(dense_r0[q], base_qvs[q], alpha, args.top_k) for q in dev_qids})
    print(f"[M9] dev R0 {r0['ndcg_at_10']:.4f}  R2 {r2['ndcg_at_10']:.4f}", flush=True)

    sweep: Dict[str, Dict] = {}
    best = ("0/0.0", -1.0)
    for k in K_GRID:
        hub = QBNorm.hubness(chassis.corpus_embeddings, bank_dev, k=k)
        for beta in BETA_GRID:
            adj = sims - beta * hub[None, :]
            dense_q = chassis.topk_from_matrix(dev_qids, adj, n_docs)
            m_dense = compute_retrieval_metrics(qrels, dense_q)
            # NB: the top-150 verified set is the *original* dense top-150 (cached); its confidences are reused.
            m_full = compute_retrieval_metrics(qrels, {q: TopKVerifier.blend(dense_q[q], base_qvs[q], alpha, n_docs) for q in dev_qids})
            key = f"{k}/{beta}"
            sweep[key] = {"k": k, "beta": beta, "dense_ndcg10": m_dense["ndcg_at_10"], "full": m_full}
            print(f"  k={k} beta={beta:<5} dense {m_dense['ndcg_at_10']:.4f}  R2+QB {m_full['ndcg_at_10']:.4f}", flush=True)
            if m_full["ndcg_at_10"] > best[1] or (m_full["ndcg_at_10"] == best[1] and beta == 0.0):
                best = (key, m_full["ndcg_at_10"])
    best_k, best_beta = sweep[best[0]]["k"], sweep[best[0]]["beta"]
    gain = best[1] - r2["ndcg_at_10"]
    decision = "ADOPT" if gain > 0 and best_beta > 0 else "REJECT"
    report = {"rung": "R4", "model": args.model, "alpha": alpha, "top_k": args.top_k, "dev_r0": r0, "dev_r2": r2,
              "sweep": sweep, "best_k": best_k, "best_beta": best_beta, "dev_r4": sweep[best[0]]["full"],
              "gain_ndcg10_vs_r2": round(gain, 5), "decision": decision}
    (DOCS_DIR / "dev_r4.json").write_text(json.dumps(report, indent=2))
    print(f"[M9] {decision}: k={best_k} beta={best_beta} dev NDCG@10 {best[1]:.4f} vs R2 {r2['ndcg_at_10']:.4f} -> docs/dev_r4.json")
    if decision == "REJECT" and not args.force_test:
        return

    import mteb

    hub_all = QBNorm.hubness(chassis.corpus_embeddings, embs, k=best_k)   # bank = all 5,000 train statements
    qb = QBNorm(hub=hub_all, beta=best_beta)
    sandbox = VerificationSandbox(default_timeout=args.timeout, reduced_timeout=0.3, workers=args.workers)
    verifier = TopKVerifier(sandbox=sandbox, default_alpha=alpha)
    model = VERASearchProtocol(chassis=chassis, verifier=verifier, verify_top_k=args.top_k, alpha=alpha, qbnorm=qb,
                               name=f"vera/VERA-R4-QBNorm-{args.model.split('/')[-1]}")
    t0 = time.time()
    result = mteb.evaluate(model, mteb.get_task("AppsRetrieval"), cache=None, overwrite_strategy="always", co2_tracker=False)
    extra = {"rung": "R4", "model": args.model, "alpha": alpha, "top_k": args.top_k, "beta": best_beta, "qb_k": best_k,
             "wall_time_s": round(time.time() - t0, 1), "timings": model.timings, "dev": sweep[best[0]]["full"]}
    save_mteb_task_result(result.task_results[0], R4_RESULTS_JSON, extra=extra)
    data = json.loads(R4_RESULTS_JSON.read_text())
    ok, errors = validate_mteb_result_schema(data)
    assert ok, errors
    row = data["scores"]["test"][0]
    print(f"\n TEST R4  NDCG@10={row['ndcg_at_10']:.4f}  MRR@10={row['mrr_at_10']:.4f}  beta={best_beta} k={best_k}")
    if args.promote:
        save_mteb_task_result(data, OFFICIAL_SUBMISSION_JSON)
    sandbox.close()


if __name__ == "__main__":
    main()
