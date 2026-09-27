"""
Mine hard negatives for R1 fine-tuning (Task 04 step 1)
=======================================================
Uses the cached base-model embeddings (corpus + 5,000 train statements):

* dense negatives: top-k non-gold corpus docs for each train-split query;
* near-duplicate cluster negatives: golds of *other* train statements whose statement
  embedding is very close to this query's (same problem re-posted, easy-to-confuse variants).

Dev-split queries are excluded. Output: ``vera/data/hard_negatives.json`` = {qid: [doc_id, ...]}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vera.chassis.baseline import DenseChassis
from vera.chassis.mine_negatives import HardNegativeMiner
from vera.chassis.preprocess import ChassisPreprocessor
from vera.data.loader import AppsRetrievalDataset

OUT = PROJECT_ROOT / "vera" / "data" / "hard_negatives.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Alibaba-NLP/gte-modernbert-base")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--cluster-threshold", type=float, default=0.85)
    args = ap.parse_args()

    ds = AppsRetrievalDataset()
    pre = ChassisPreprocessor()
    chassis = DenseChassis(model_name=args.model, batch_size=8)
    chassis.index_corpus(pre.process_corpus(ds.get_corpus()))
    dev = set(ds.get_or_create_dev_split()["dev_query_ids"])
    train_ids = [q for q in sorted(ds.train_qrels) if q not in dev]
    all_train = pre.process_queries({q: ds.queries_dict[q] for q in sorted(ds.train_qrels)})
    qids, embs = chassis.encode_queries(all_train, tag="train_queries")
    pos = {q: i for i, q in enumerate(qids)}
    q_embs = embs[[pos[q] for q in train_ids]]

    sims = chassis.similarity(q_embs)
    dense = chassis.topk_from_matrix(train_ids, sims, args.top_k + 1)
    miner = HardNegativeMiner(top_k_dense=args.top_k)
    negatives = miner.mine_dense_negatives({q: "" for q in train_ids}, ds.train_qrels, dense)
    clusters = miner.mine_cluster_negatives({q: "" for q in train_ids}, q_embs, train_ids, cluster_threshold=args.cluster_threshold)
    n_cluster = 0
    for q, near in clusters.items():
        for other in near:
            for gold in ds.train_qrels.get(other, {}):
                if gold not in ds.train_qrels[q] and gold not in negatives[q]:
                    negatives[q].insert(0, gold)  # cluster negatives first: hardest
                    n_cluster += 1
    OUT.write_text(json.dumps(negatives))
    print(f"mined negatives for {len(negatives)} train queries ({n_cluster} near-duplicate cluster negatives) -> {OUT}")


if __name__ == "__main__":
    main()
