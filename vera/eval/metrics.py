"""
Retrieval metrics for the dev split (the test split is scored by ``mteb.evaluate`` itself).

The formulas follow pytrec_eval / MTEB: binary relevance, NDCG with log2 discount,
MRR@k, recall@k, MAP@k with the min(k, |relevant|) normalizer.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping

DEFAULT_K_VALUES = (1, 3, 5, 10, 20, 100, 1000)


def compute_retrieval_metrics(
    qrels: Mapping[str, Mapping[str, int]],
    results: Mapping[str, Mapping[str, float]],
    k_values: Iterable[int] = DEFAULT_K_VALUES,
) -> Dict[str, float]:
    """Mean NDCG/MRR/recall/MAP at each k over all queries in ``qrels``."""
    ks = list(k_values)
    acc = {f"{m}_at_{k}": 0.0 for k in ks for m in ("ndcg", "mrr", "recall", "map")}
    n = 0
    for qid, gold in qrels.items():
        gold_ids = {d for d, rel in gold.items() if rel > 0}
        if not gold_ids:
            continue
        n += 1
        ranked = [d for d, _ in sorted(results.get(qid, {}).items(), key=lambda kv: kv[1], reverse=True)]
        for k in ks:
            top = ranked[:k]
            hits = [1 if d in gold_ids else 0 for d in top]
            num_hits = sum(hits)
            acc[f"recall_at_{k}"] += num_hits / len(gold_ids)
            first = next((i + 1 for i, h in enumerate(hits) if h), None)
            acc[f"mrr_at_{k}"] += (1.0 / first) if first else 0.0
            if num_hits:
                running, precs = 0, []
                for i, h in enumerate(hits):
                    if h:
                        running += 1
                        precs.append(running / (i + 1))
                acc[f"map_at_{k}"] += sum(precs) / min(k, len(gold_ids))
            dcg = sum(1.0 / math.log2(i + 2) for i, h in enumerate(hits) if h)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(gold_ids))))
            acc[f"ndcg_at_{k}"] += dcg / idcg if idcg else 0.0
    return {key: round(v / max(1, n), 5) for key, v in acc.items()}


def rank_of_gold(qrels: Mapping[str, Mapping[str, int]], results: Mapping[str, Mapping[str, float]]) -> Dict[str, int]:
    """1-based rank of the (single) gold doc per query, or 0 when it is not in the result list."""
    out: Dict[str, int] = {}
    for qid, gold in qrels.items():
        gold_ids = {d for d, rel in gold.items() if rel > 0}
        ranked = [d for d, _ in sorted(results.get(qid, {}).items(), key=lambda kv: kv[1], reverse=True)]
        out[qid] = next((i + 1 for i, d in enumerate(ranked) if d in gold_ids), 0)
    return out
