"""
VERA AbsEncoder Fallback Interface
===================================
Fallback pure-embedding encoder interface in case hackathon organizers
require an AbsEncoder subclass rather than SearchProtocol.
"""

from __future__ import annotations

import numpy as np
from typing import Any, List, Optional, Union

try:
    from mteb.encoder_interface import AbsEncoder, ModelMeta
    HAS_ABS_ENCODER = True
except ImportError:
    HAS_ABS_ENCODER = False
    AbsEncoder = object
    ModelMeta = None


class VERAAbsEncoder(AbsEncoder):
    """
    Encoder-only fallback interface adhering to MTEB AbsEncoder specs.
    Encodes text into dense vectors using the fine-tuned chassis.
    """

    def __init__(
        self,
        model_name_or_path: str = "Alibaba-NLP/gte-modernbert-base",
        embed_dim: int = 768,
        max_length: int = 8192,
        encode_fn: Optional[Any] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.encode_fn = encode_fn

        if ModelMeta is not None:
            self.mteb_model_meta = ModelMeta(
                name=f"VERA-Chassis-{model_name_or_path.split('/')[-1]}",
                revision="1.0.0",
                release_date="2026-09-27",
                languages=["python"],
                loader=None,
                n_parameters=None,
                memory_usage_mb=512,
                max_tokens=self.max_length,
                embed_dim=self.embed_dim,
                license="mit",
                open_weights=True,
                similarity_fn_name="cosine",
                use_instructions=False,
                framework=["PyTorch", "ONNX"],
            )

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encodes arbitrary texts into normalized float32 vectors."""
        if self.encode_fn is not None:
            return self.encode_fn(sentences, batch_size=batch_size, **kwargs)

        # Fallback deterministic pseudo-embeddings for testing
        num_items = len(sentences)
        np.random.seed(42)
        vecs = np.random.randn(num_items, self.embed_dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-12)

    def encode_queries(self, queries: List[str], **kwargs: Any) -> np.ndarray:
        return self.encode(queries, **kwargs)

    def encode_corpus(self, corpus: Union[List[Dict[str, str]], List[str]], **kwargs: Any) -> np.ndarray:
        if isinstance(corpus, list) and len(corpus) > 0 and isinstance(corpus[0], dict):
            texts = [doc.get("text", "") for doc in corpus]
        else:
            texts = [str(doc) for doc in corpus]
        return self.encode(texts, **kwargs)
