"""
VERA Dense Chassis (L1)
=======================
Encodes problem statements and boilerplate-stripped solutions with a dense
bi-encoder (default ``Alibaba-NLP/gte-modernbert-base`` at its full 8192-token
context) and performs brute-force cosine search on CPU (8,765 docs fit in RAM;
no ANN index needed).

Backends
--------
``st``     sentence-transformers model (the real R0 chassis).
``tfidf``  lexical TF-IDF reference used **only** as an explicit ablation row.

There is deliberately *no* silent fallback: if the transformer cannot be loaded
the call raises, so a milestone JSON can never be produced by accident with a
different model than its ``model_name`` claims.

Embeddings are cached under ``vera/chassis/cache`` keyed by a fingerprint of
(model, backend, max_seq_length, texts) so the R2/R4 stages reuse them.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

CHASSIS_CACHE_DIR = Path(__file__).resolve().parent / "cache"
DEFAULT_MODEL = "Alibaba-NLP/gte-modernbert-base"


def _fingerprint(*parts: str) -> str:
    h = hashlib.md5()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\x00")
    return h.hexdigest()[:12]


class DenseChassis:
    """Dense embedding, caching and CPU similarity search."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        backend: str = "st",
        max_seq_length: int = 8192,
        batch_size: int = 8,
        cache_dir: Optional[Path] = None,
        num_threads: Optional[int] = None,
        show_progress: bool = True,
    ):
        if backend not in ("st", "tfidf"):
            raise ValueError(f"unknown backend {backend!r}")
        self.model_name = model_name
        self.backend = backend
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else CHASSIS_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.num_threads = num_threads or (os.cpu_count() or 4)
        self.show_progress = show_progress

        self.doc_ids: List[str] = []
        self.corpus_embeddings: Optional[np.ndarray] = None
        self._model = None
        self._vectorizer = None
        self._corpus_sparse = None

    # ------------------------------------------------------------------ #
    @property
    def tag(self) -> str:
        return f"{self.backend}-{self.model_name.split('/')[-1]}-L{self.max_seq_length}"

    def _load_model(self):
        if self._model is not None or self.backend != "st":
            return self._model
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(self.num_threads)
        t0 = time.time()
        model = SentenceTransformer(self.model_name, device="cpu")
        model.max_seq_length = self.max_seq_length
        self._model = model
        print(f"[Dense Chassis] Loaded {self.model_name} (max_seq_length={model.max_seq_length}) in {time.time() - t0:.1f}s")
        return model

    # ------------------------------------------------------------------ #
    def encode(self, texts: Sequence[str], desc: str = "texts") -> np.ndarray:
        """Encode texts to L2-normalized float32 vectors (no caching)."""
        texts = [t if isinstance(t, str) and t.strip() else " " for t in texts]
        if self.backend == "tfidf":
            if self._vectorizer is None:
                raise RuntimeError("tfidf backend: call index_corpus() first")
            mat = self._vectorizer.transform(texts)
            return mat  # sparse, already l2-normalised by the vectorizer
        model = self._load_model()
        t0 = time.time()
        emb = model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=self.show_progress and len(texts) > 64,
        )
        dt = time.time() - t0
        print(f"[Dense Chassis] Encoded {len(texts)} {desc} in {dt:.1f}s ({len(texts) / max(dt, 1e-9):.2f}/s)")
        return np.asarray(emb, dtype=np.float32)

    def encode_cached(self, texts: Sequence[str], tag: str, force: bool = False) -> np.ndarray:
        """Encode with an on-disk cache keyed by model + texts."""
        fp = _fingerprint(self.tag, str(len(texts)), *texts)
        path = self.cache_dir / f"{self.tag}_{fp}.npy"
        if not force:
            # The fingerprint covers model + texts, so any file with this fingerprint is valid whatever its prefix.
            hits = [path] if path.exists() else sorted(self.cache_dir.glob(f"*{self.tag}_{fp}.npy"))
            if hits:
                emb = np.load(hits[0])
                print(f"[Dense Chassis] Loaded cached {tag} embeddings {emb.shape} from {hits[0].name}")
                return emb
        emb = self.encode(texts, desc=tag)
        if self.backend == "st":
            np.save(path, emb)
            print(f"[Dense Chassis] Cached {tag} embeddings -> {path.name}")
        return emb

    # ------------------------------------------------------------------ #
    def index_corpus(self, corpus: Dict[str, str], force_recompute: bool = False) -> None:
        """Encode (or load cached) embeddings for the whole corpus."""
        self.doc_ids = sorted(corpus.keys())
        texts = [corpus[d] for d in self.doc_ids]
        if self.backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectorizer = TfidfVectorizer(
                token_pattern=r"[A-Za-z_][A-Za-z0-9_]*|\d+|\S", ngram_range=(1, 2), sublinear_tf=True,
                min_df=1, max_features=300000, norm="l2", lowercase=True,
            )
            self._corpus_sparse = self._vectorizer.fit_transform(texts)
            self.corpus_embeddings = None
            print(f"[Dense Chassis] TF-IDF indexed {len(texts)} docs, vocab={len(self._vectorizer.vocabulary_)}")
            return
        self.corpus_embeddings = self.encode_cached(texts, tag="corpus", force=force_recompute)

    def encode_queries(self, queries: Dict[str, str], tag: str = "queries") -> Tuple[List[str], np.ndarray]:
        qids = list(queries.keys())
        embs = self.encode_cached([queries[q] for q in qids], tag=tag) if self.backend == "st" else self.encode([queries[q] for q in qids])
        return qids, embs

    def similarity(self, query_embs) -> np.ndarray:
        """Dense (n_queries x n_docs) cosine similarity matrix."""
        if self.backend == "tfidf":
            return (query_embs @ self._corpus_sparse.T).toarray().astype(np.float32)
        if self.corpus_embeddings is None:
            raise RuntimeError("Corpus has not been indexed! Call index_corpus() first.")
        return np.matmul(query_embs, self.corpus_embeddings.T)

    def search(self, queries: Dict[str, str], top_k: int = 150, tag: str = "queries") -> Dict[str, Dict[str, float]]:
        """Return ``{qid: {doc_id: score}}`` for the top_k docs per query."""
        qids, q_embs = self.encode_queries(queries, tag=tag)
        sims = self.similarity(q_embs)
        return self.topk_from_matrix(qids, sims, top_k)

    def topk_from_matrix(self, qids: List[str], sims: np.ndarray, top_k: int) -> Dict[str, Dict[str, float]]:
        results: Dict[str, Dict[str, float]] = {}
        k = min(top_k, sims.shape[1])
        for i, qid in enumerate(qids):
            row = sims[i]
            idx = np.argpartition(row, -k)[-k:] if k < len(row) else np.arange(len(row))
            idx = idx[np.argsort(-row[idx])]
            results[qid] = {self.doc_ids[j]: float(row[j]) for j in idx}
        return results
