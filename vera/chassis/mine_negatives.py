"""
VERA Hard Negative Miner
========================
Mines hard negatives for contrastive encoder fine-tuning:
1. Dense non-gold negatives: Top-k non-gold candidates retrieved by base model.
2. Near-duplicate problem cluster negatives: Solutions to similar but distinct problems.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np


class HardNegativeMiner:
    """Mines informative negative solutions for training pairs."""

    def __init__(self, top_k_dense: int = 15):
        self.top_k_dense = top_k_dense

    def mine_dense_negatives(
        self,
        train_queries: Dict[str, str],
        train_qrels: Dict[str, Dict[str, int]],
        candidate_scores: Dict[str, Dict[str, float]],
    ) -> Dict[str, List[str]]:
        """
        Extracts top-k non-gold document IDs per query based on dense similarity scores.
        """
        hard_negatives: Dict[str, List[str]] = {}

        for qid in train_queries:
            gold_ids = set(train_qrels.get(qid, {}).keys())
            preds = candidate_scores.get(qid, {})

            # Sort descending by score
            sorted_candidates = sorted(preds.items(), key=lambda x: x[1], reverse=True)
            non_golds = [cid for cid, _ in sorted_candidates if cid not in gold_ids]

            hard_negatives[qid] = non_golds[: self.top_k_dense]

        return hard_negatives

    def mine_cluster_negatives(
        self,
        train_queries: Dict[str, str],
        query_embeddings: np.ndarray,
        qids: List[str],
        cluster_threshold: float = 0.85,
    ) -> Dict[str, List[str]]:
        """
        Identifies queries with high semantic similarity (near duplicates)
        and exchanges their solutions as hard adversarial negatives.
        """
        cluster_negatives: Dict[str, List[str]] = {qid: [] for qid in qids}
        sim_matrix = np.matmul(query_embeddings, query_embeddings.T)

        for i, qid in enumerate(qids):
            # Find distinct queries with high similarity
            sims = sim_matrix[i]
            near_dup_indices = np.where((sims >= cluster_threshold) & (sims < 0.999))[0]
            cluster_qids = [qids[idx] for idx in near_dup_indices[:5]]
            cluster_negatives[qid] = cluster_qids

        return cluster_negatives
