"""
VERA SearchProtocol Submission Interface
========================================
Implements MTEB v2 SearchProtocol interface for custom retrieval pipelines.
Allows end-to-end integration of dense embedding, dynamic verification,
and calibrated score boosting within mteb.evaluate().
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

# Attempt import of MTEB ModelMeta if available
try:
    from mteb.models.overview import ModelMeta
    HAS_MTEB_META = True
except ImportError:
    try:
        from mteb.encoder_interface import ModelMeta
        HAS_MTEB_META = True
    except ImportError:
        HAS_MTEB_META = False
        ModelMeta = None


class DummyModelMeta:
    """Mock ModelMeta to satisfy MTEB inspection if ModelMeta is unavailable."""
    def __init__(self, name: str = "VERA-SearchProtocol-Pipeline"):
        self.name = name
        self.revision = "1.0.0"
        self.release_date = "2026-09-27"
        self.languages = ["python"]
        self.loader = None
        self.n_parameters = None
        self.memory_usage_mb = 512
        self.max_tokens = 8192
        self.embed_dim = 768
        self.license = "mit"
        self.open_weights = True
        self.similarity_fn_name = "cosine"
        self.use_instructions = False
        self.framework = ["PyTorch", "ONNX"]


class VERASearchProtocol:
    """
    Primary submission class implementing MTEB v2 SearchProtocol.
    Exposes .index() and .search() interfaces.
    """

    def __init__(
        self,
        name: str = "VERA-SearchProtocol-Pipeline",
        dense_backend: Optional[Callable[[Dict[str, str], int], Dict[str, Dict[str, float]]]] = None,
        verifier_backend: Optional[Callable[[str, List[str]], Dict[str, float]]] = None,
    ):
        self.name = name
        self.dense_backend = dense_backend
        self.verifier_backend = verifier_backend
        self.corpus_index: Dict[str, str] = {}

        # Configure ModelMeta for MTEB v2
        if HAS_MTEB_META and ModelMeta is not None:
            try:
                self.mteb_model_meta = ModelMeta(
                    name=self.name,
                    revision="1.0.0",
                    release_date="2026-09-27",
                    languages=["python"],
                    loader=None,
                    n_parameters=None,
                    memory_usage_mb=512,
                    max_tokens=8192,
                    embed_dim=768,
                    license="mit",
                    open_weights=True,
                    similarity_fn_name="cosine",
                    use_instructions=False,
                    framework=["PyTorch", "ONNX"],
                )
            except Exception:
                self.mteb_model_meta = DummyModelMeta(name=self.name)
        else:
            self.mteb_model_meta = DummyModelMeta(name=self.name)

    def index(self, corpus: Union[Dict[str, Dict[str, str]], List[Dict[str, str]], Dict[str, str]], **kwargs: Any) -> None:
        """
        Indexes the candidate corpus documents.
        Corpus format from MTEB can be:
          - dict[doc_id, {"text": "..."}]
          - dict[doc_id, "text"]
          - list[{"id": doc_id, "text": "..."}]
        """
        self.corpus_index.clear()

        if isinstance(corpus, dict):
            for doc_id, item in corpus.items():
                if isinstance(item, dict):
                    self.corpus_index[str(doc_id)] = item.get("text", "")
                else:
                    self.corpus_index[str(doc_id)] = str(item)
        elif isinstance(corpus, list):
            for item in corpus:
                doc_id = str(item.get("id", item.get("_id", "")))
                self.corpus_index[doc_id] = item.get("text", "")

        print(f"[{self.name}] Indexed {len(self.corpus_index)} corpus documents.")

    def search(
        self,
        queries: Union[Dict[str, str], List[str]],
        top_k: int = 100,
        **kwargs: Any,
    ) -> Dict[str, Dict[str, float]]:
        """
        Executes search for input queries and returns score dictionary:
        {query_id: {corpus_id: score, ...}, ...}
        """
        # Normalize queries format
        norm_queries: Dict[str, str] = {}
        if isinstance(queries, dict):
            norm_queries = {str(qid): str(text) for qid, text in queries.items()}
        elif isinstance(queries, list):
            norm_queries = {str(i): str(text) for i, text in enumerate(queries)}

        # If a custom dense backend is provided, execute it
        if self.dense_backend is not None:
            dense_results = self.dense_backend(norm_queries, top_k)
        else:
            # Fallback lightweight term-overlap or stub ranking for harness testing
            dense_results = self._fallback_stub_search(norm_queries, top_k)

        # If verification backend is configured, apply reranking/boosting
        if self.verifier_backend is not None:
            final_results: Dict[str, Dict[str, float]] = {}
            for qid, q_text in norm_queries.items():
                q_candidates = dense_results.get(qid, {})
                boosts = self.verifier_backend(q_text, list(q_candidates.keys()))
                # Apply additive blend
                combined = {
                    cid: float(q_candidates[cid]) + float(boosts.get(cid, 0.0))
                    for cid in q_candidates
                }
                final_results[qid] = combined
            return final_results

        return dense_results

    def _fallback_stub_search(self, queries: Dict[str, str], top_k: int) -> Dict[str, Dict[str, float]]:
        """
        Deterministic lightweight search stub for proof-of-life harness testing.
        Uses exact token overlap without requiring GPU or large weights.
        """
        results: Dict[str, Dict[str, float]] = {}
        corpus_items = list(self.corpus_index.items())

        for qid, q_text in queries.items():
            q_words = set(q_text.lower().split()[:50])
            scores = []
            for doc_id, doc_text in corpus_items[: min(top_k * 3, len(corpus_items))]:
                doc_words = set(doc_text.lower().split()[:100])
                overlap = len(q_words.intersection(doc_words))
                score = overlap / max(1, len(q_words))
                scores.append((doc_id, score))

            # Sort and take top_k
            scores.sort(key=lambda x: x[1], reverse=True)
            results[qid] = {cid: float(s) for cid, s in scores[:top_k]}

        return results
