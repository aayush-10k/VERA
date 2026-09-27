"""
VERA SearchProtocol Submission Interface
========================================
Implements the MTEB v2 ``SearchProtocol`` (``index`` / ``search`` / ``mteb_model_meta``)
so the whole pipeline (dense chassis -> optional top-K verification boost ->
optional QB-Norm demotion) runs *inside* ``mteb.evaluate`` and the official
``TaskResult`` JSON is produced by MTEB itself.

Hard rule: this class only ever reads the ``id`` and ``text`` columns of the
corpus/query datasets. ``partition`` and ``meta_information`` are never touched.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from vera.chassis.baseline import DenseChassis
from vera.chassis.preprocess import ChassisPreprocessor

try:  # ModelMeta lives in mteb.models.model_meta in mteb>=2
    from mteb.models.model_meta import ModelMeta
except Exception:  # pragma: no cover
    ModelMeta = None  # type: ignore[assignment]


def _ids_and_texts(data: Any) -> Tuple[List[str], List[str]]:
    """Accept an HF ``Dataset`` (id/text columns), ``{id: text}``, ``{id: {"text": ..}}`` or ``[{"id":..,"text":..}]``."""
    if hasattr(data, "column_names"):  # datasets.Dataset
        ids = [str(x) for x in data["id"]]
        texts = [str(x) for x in data["text"]]
        return ids, texts
    if isinstance(data, Mapping):
        ids, texts = [], []
        for k, v in data.items():
            ids.append(str(k))
            texts.append(str(v.get("text", "")) if isinstance(v, Mapping) else str(v))
        return ids, texts
    if isinstance(data, Sequence):
        ids = [str(item.get("id", item.get("_id", i))) for i, item in enumerate(data)]
        texts = [str(item.get("text", "")) for item in data]
        return ids, texts
    raise TypeError(f"Unsupported corpus/query container: {type(data)}")


def build_model_meta(name: str, revision: str = "1.0.0", max_tokens: int = 8192, embed_dim: int = 768) -> Any:
    """A ModelMeta that satisfies mteb's validators (or a plain namespace when mteb is unavailable)."""
    if ModelMeta is None:  # pragma: no cover
        return type("Meta", (), {"name": name, "revision": revision})()
    return ModelMeta(
        loader=None,
        name=name,
        revision=revision,
        release_date="2026-09-27",
        languages=["eng-Latn", "python-Code"],
        n_parameters=149_000_000,
        memory_usage_mb=570,
        max_tokens=max_tokens,
        embed_dim=embed_dim,
        license="mit",
        open_weights=True,
        public_training_code=None,
        public_training_data=None,
        framework=["Sentence Transformers", "PyTorch"],
        similarity_fn_name="cosine",
        use_instructions=False,
        training_datasets=None,
    )


class VERASearchProtocol:
    """MTEB v2 SearchProtocol: dense retrieval + (optional) verification boost + (optional) QB-Norm.

    Parameters
    ----------
    chassis : DenseChassis
        The dense encoder/index (R0/R1).
    verifier : TopKVerifier | None
        When given, the top ``verify_top_k`` dense candidates of each query are executed on the
        query's worked examples and re-scored (R2).
    qbnorm : callable | None
        ``fn(query_ids, sims) -> sims`` hook applied to the dense similarity matrix (R4).
    name : str
        Reported as the model name in the TaskResult (``org/model`` form).
    """

    def __init__(
        self,
        chassis: DenseChassis,
        verifier: Optional[Any] = None,
        verify_top_k: int = 150,
        alpha: Optional[float] = None,
        qbnorm: Optional[Any] = None,
        name: str = "vera/VERA-R0-gte-modernbert-base",
        remove_examples_from_query: bool = False,
        query_tag: str = "queries",
    ):
        self.chassis = chassis
        self.verifier = verifier
        self.verify_top_k = verify_top_k
        self.alpha = alpha
        self.qbnorm = qbnorm
        self.name = name
        self.remove_examples_from_query = remove_examples_from_query
        self.query_tag = query_tag
        self.preprocessor = ChassisPreprocessor()
        self.raw_corpus: Dict[str, str] = {}
        self.timings: Dict[str, float] = {}
        self._meta = build_model_meta(name, max_tokens=chassis.max_seq_length)

    # ------------------------------------------------------------------ #
    @property
    def mteb_model_meta(self):
        return self._meta

    def index(self, corpus: Any, **kwargs: Any) -> None:
        t0 = time.time()
        ids, texts = _ids_and_texts(corpus)
        self.raw_corpus = dict(zip(ids, texts))
        stripped = self.preprocessor.process_corpus(self.raw_corpus)
        self.chassis.index_corpus(stripped)
        self.timings["index_s"] = time.time() - t0
        print(f"[{self.name}] Indexed {len(ids)} docs in {self.timings['index_s']:.1f}s")

    def search(self, queries: Any, *, top_k: int = 1000, **kwargs: Any) -> Dict[str, Dict[str, float]]:
        t0 = time.time()
        qids, texts = _ids_and_texts(queries)
        raw_queries = dict(zip(qids, texts))
        clean = self.preprocessor.process_queries(raw_queries, remove_examples=self.remove_examples_from_query)

        split = kwargs.get("hf_split") or "test"
        q_order, q_embs = self.chassis.encode_queries(clean, tag=f"{self.query_tag}-{split}")
        sims = self.chassis.similarity(q_embs)
        if self.qbnorm is not None:
            sims = self.qbnorm(q_order, sims)
        k = max(top_k, self.verify_top_k if self.verifier is not None else 0)
        dense = self.chassis.topk_from_matrix(q_order, sims, k)
        self.timings["dense_s"] = time.time() - t0

        if self.verifier is None:
            return {q: dict(list(r.items())[:top_k]) for q, r in dense.items()}

        t1 = time.time()
        out: Dict[str, Dict[str, float]] = {}
        for i, qid in enumerate(q_order):
            reranked = self.verifier.rerank_query(
                query_text=raw_queries[qid],
                dense_scores=dense[qid],
                corpus_dict=self.raw_corpus,
                top_k=self.verify_top_k,
                alpha=self.alpha,
            )
            out[qid] = dict(list(reranked.items())[:top_k])
            if (i + 1) % 200 == 0 or i + 1 == len(q_order):
                print(f"[{self.name}] verified {i + 1}/{len(q_order)} queries ({time.time() - t1:.0f}s)", flush=True)
        self.timings["verify_s"] = time.time() - t1
        return out
