"""
QB-Norm content-based demotion (Rung R4)
========================================
Querybank normalisation with the 5,000 public *train* problem statements as the bank.

Why it works here: every train-partition solution in the corpus is the gold of a train
statement that sits in the bank, so such a document has an unusually strong affinity to
the bank ("hubness"). Demoting by that affinity captures the same effect as reading the
forbidden ``partition`` tag, but purely from public text content:

    hub(d)          = mean of the top-k cosine similarities between doc d and the bank
    S_qbnorm(q, d)  = S(q, d) - beta * hub(d)

beta and k are fit on the dev split. When fitting on dev the 500 dev statements are
*removed* from the bank (their golds would otherwise be demoted by their own statement,
which does not happen for test queries).

The classic dynamic-inverted-softmax variant (Bogolin et al., 2022) is available as
``mode="dis"`` for the ablation table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


@dataclass
class QBNorm:
    """Callable hook ``(query_ids, sims) -> sims`` for ``VERASearchProtocol``."""

    hub: np.ndarray                # (n_docs,) affinity of each corpus doc to the bank
    beta: float = 0.0
    mode: str = "topk_mean"

    def __call__(self, query_ids: Sequence[str], sims: np.ndarray) -> np.ndarray:
        if self.beta == 0.0:
            return sims
        return sims - self.beta * self.hub[None, :]

    @staticmethod
    def hubness(doc_embs: np.ndarray, bank_embs: np.ndarray, k: int = 3, chunk: int = 2048) -> np.ndarray:
        """Mean of the top-k similarities of every doc to the bank (both inputs L2-normalised)."""
        k = max(1, min(k, bank_embs.shape[0]))
        out = np.empty(doc_embs.shape[0], dtype=np.float32)
        for start in range(0, doc_embs.shape[0], chunk):
            block = doc_embs[start:start + chunk] @ bank_embs.T          # (chunk, n_bank)
            if k == 1:
                out[start:start + chunk] = block.max(axis=1)
            else:
                part = np.partition(block, -k, axis=1)[:, -k:]
                out[start:start + chunk] = part.mean(axis=1)
        return out

    @classmethod
    def fit_hub(cls, doc_embs: np.ndarray, bank_embs: np.ndarray, k: int = 3, beta: float = 0.0) -> "QBNorm":
        return cls(hub=cls.hubness(doc_embs, bank_embs, k=k), beta=beta, mode="topk_mean")


def dis_normalize(sims: np.ndarray, bank_sims: np.ndarray, temperature: float = 1 / 0.02) -> np.ndarray:
    """Dynamic inverted softmax (QB-Norm paper): normalise each doc column by its softmax mass over the bank.

    ``sims``: (n_q, n_docs) test-query similarities; ``bank_sims``: (n_bank, n_docs) bank-query similarities.
    Only docs that are the top-1 of some test query get normalised (the 'dynamic' part).
    """
    exp_q = np.exp(temperature * sims)
    exp_b = np.exp(temperature * bank_sims)
    denom = exp_b.sum(axis=0, keepdims=True)                              # (1, n_docs)
    normalised = exp_q / np.maximum(denom, 1e-30)
    top1 = sims.argmax(axis=1)
    mask = np.zeros(sims.shape[1], dtype=bool)
    mask[np.unique(top1)] = True
    out = sims.copy()
    out[:, mask] = normalised[:, mask] * sims[:, mask].max() / max(normalised[:, mask].max(), 1e-30)
    return out
