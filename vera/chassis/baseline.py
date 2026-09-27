"""
VERA Dense Chassis Baseline (R0)
================================
Encodes problem statements and boilerplate-stripped solutions using 
dense bi-encoder architecture with 8192 token context.
Supports SentenceTransformers with resilient fallback to high-speed TF-IDF/SVD
vectorization for robust, rate-limit-immune CPU execution.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

CHASSIS_CACHE_DIR = Path(__file__).resolve().parent / "cache"


class DenseChassis:
    """Manages dense embeddings, caching, and CPU similarity search."""

    def __init__(
        self,
        model_name: str = "Alibaba-NLP/gte-modernbert-base",
        cache_dir: Optional[Path] = None,
        embed_dim: int = 768,
        batch_size: int = 32,
        use_fallback: bool = False,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir or CHASSIS_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.use_fallback = use_fallback

        self.doc_ids: List[str] = []
        self.corpus_embeddings: Optional[np.ndarray] = None
        self._model = None
        self._vectorizer = None

    def _get_cache_path(self, corpus_hash: str) -> Path:
        model_hash = hashlib.md5(self.model_name.encode()).hexdigest()[:8]
        return self.cache_dir / f"corpus_{model_hash}_{corpus_hash}.npz"

    def _load_model(self):
        """Attempts to load transformer encoder; falls back cleanly on rate limit or offline."""
        if self._model is not None or self.use_fallback:
            return

        try:
            from sentence_transformers import SentenceTransformer
            print(f"[Dense Chassis] Loading encoder: {self.model_name}...")
            # Use small timeout or local check
            self._model = SentenceTransformer(self.model_name, device="cpu")
        except Exception as e:
            print(f"[Dense Chassis] Transformer download unavailable or throttled ({e}).")
            print("[Dense Chassis] Switching to resilient high-speed TF-IDF dense chassis.")
            self.use_fallback = True

    def fit_fallback_vectorizer(self, corpus_texts: List[str]):
        """Fits TF-IDF vectorizer over the corpus vocabulary."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(
            max_features=self.embed_dim,
            ngram_range=(1, 2),
            stop_words="english",
            norm="l2",
        )
        self._vectorizer.fit(corpus_texts)

    def encode_texts(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """Encodes list of strings to normalized float32 embeddings."""
        self._load_model()
        if self._model is not None and not self.use_fallback:
            try:
                embeddings = self._model.encode(
                    texts,
                    batch_size=self.batch_size,
                    show_progress_bar=len(texts) > 200,
                    normalize_embeddings=normalize,
                )
                return np.asarray(embeddings, dtype=np.float32)
            except Exception as e:
                print(f"[Dense Chassis] SentenceTransformer encode failed ({e}), falling back.")
                self.use_fallback = True

        # High-speed TF-IDF fallback vectorization
        if self._vectorizer is None:
            self.fit_fallback_vectorizer(texts)

        sparse_matrix = self._vectorizer.transform(texts)
        dense = sparse_matrix.toarray().astype(np.float32)

        # Pad to embed_dim if needed
        if dense.shape[1] < self.embed_dim:
            pad = np.zeros((dense.shape[0], self.embed_dim - dense.shape[1]), dtype=np.float32)
            dense = np.hstack([dense, pad])

        if normalize:
            norms = np.linalg.norm(dense, axis=1, keepdims=True)
            dense = dense / np.maximum(norms, 1e-12)

        return dense

    def index_corpus(
        self,
        corpus: Dict[str, str],
        force_recompute: bool = False,
    ) -> None:
        """
        Encodes and caches all corpus documents.
        Uses cached numpy arrays if available to save recomputation.
        """
        self.doc_ids = sorted(list(corpus.keys()))
        texts = [corpus[doc_id] for doc_id in self.doc_ids]

        corpus_hash = hashlib.md5("".join(self.doc_ids[:50]).encode()).hexdigest()[:8]
        cache_path = self._get_cache_path(corpus_hash)

        if not force_recompute and cache_path.exists():
            print(f"[Dense Chassis] Loading cached corpus embeddings from {cache_path}...")
            data = np.load(cache_path, allow_pickle=True)
            self.corpus_embeddings = data["embeddings"]
            self.doc_ids = list(data["doc_ids"])
            print(f"[Dense Chassis] Loaded {len(self.doc_ids)} embeddings of shape {self.corpus_embeddings.shape}.")
            return

        print(f"[Dense Chassis] Encoding {len(texts)} corpus documents on CPU...")
        if self.use_fallback or self._vectorizer is None:
            self.fit_fallback_vectorizer(texts)

        self.corpus_embeddings = self.encode_texts(texts, normalize=True)

        np.savez_compressed(
            cache_path,
            embeddings=self.corpus_embeddings,
            doc_ids=np.array(self.doc_ids),
        )
        print(f"[Dense Chassis] Saved embeddings to {cache_path} ({cache_path.stat().st_size / 1024:.1f} KB).")

    def search(
        self,
        queries: Dict[str, str],
        top_k: int = 150,
    ) -> Dict[str, Dict[str, float]]:
        """
        Runs matrix multiplication search against indexed corpus:
        Scores = QueryEmbeddings @ CorpusEmbeddings.T
        """
        if self.corpus_embeddings is None:
            raise RuntimeError("Corpus has not been indexed! Call index_corpus() first.")

        q_ids = list(queries.keys())
        q_texts = [queries[qid] for qid in q_ids]

        print(f"[Dense Chassis] Encoding {len(q_texts)} queries...")
        query_embeddings = self.encode_texts(q_texts, normalize=True)

        print(f"[Dense Chassis] Computing brute-force CPU similarity ({len(q_ids)} queries x {len(self.doc_ids)} docs)...")
        similarity_matrix = np.matmul(query_embeddings, self.corpus_embeddings.T)

        results: Dict[str, Dict[str, float]] = {}
        for i, qid in enumerate(q_ids):
            scores = similarity_matrix[i]
            if top_k < len(scores):
                top_indices = np.argpartition(scores, -top_k)[-top_k:]
                sorted_top = top_indices[np.argsort(-scores[top_indices])]
            else:
                sorted_top = np.argsort(-scores)

            results[qid] = {
                self.doc_ids[idx]: float(scores[idx])
                for idx in sorted_top[:top_k]
            }

        return results
