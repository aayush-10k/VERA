"""
VERA encoder-only fallback (kept warm in case the organizers rule "encoder only")
==============================================================================
MTEB v2 accepts a ``sentence_transformers.SentenceTransformer`` directly in
``mteb.evaluate`` (it is wrapped into an ``EncoderProtocol`` internally), so the
encoder-only path is simply the chassis model itself with the 8192 context set.

``build_encoder_only_model()`` returns that object; ``VERAAbsEncoder`` is a thin
subclass kept for API compatibility with earlier code/tests.
"""

from __future__ import annotations

from typing import Any, Optional

DEFAULT_MODEL = "Alibaba-NLP/gte-modernbert-base"


def build_encoder_only_model(model_name: str = DEFAULT_MODEL, max_seq_length: int = 8192, device: str = "cpu") -> Any:
    """SentenceTransformer chassis ready to be passed to ``mteb.evaluate`` as an encoder."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_length
    return model


class VERAAbsEncoder:
    """Lazy holder for the encoder-only fallback; ``.model`` is the SentenceTransformer."""

    def __init__(self, model_name_or_path: str = DEFAULT_MODEL, max_length: int = 8192, device: str = "cpu"):
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.device = device
        self._model: Optional[Any] = None

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = build_encoder_only_model(self.model_name_or_path, self.max_length, self.device)
        return self._model

    def encode(self, sentences, batch_size: int = 8, **kwargs):
        return self.model.encode(list(sentences), batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True)

    encode_queries = encode

    def encode_corpus(self, corpus, **kwargs):
        texts = [d.get("text", "") if isinstance(d, dict) else str(d) for d in corpus]
        return self.encode(texts, **kwargs)
