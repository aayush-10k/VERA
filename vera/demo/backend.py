"""
Demo backend (UI-agnostic)
==========================
Everything the Gradio page shows is produced here so it can be exercised headlessly:

* ``retrieve(query)`` — dense candidates -> top-K verification -> badges
  (``PASSED 2/2 examples``, dense rank -> final rank, rarity confidence).
* ``version_chain(path)`` / ``rank_chain(query, path)`` — version lineage from the
  Stage-2 store with ``behaviorally unchanged`` certification and working-first ranking.
* ``ingest(source)`` — folder snapshot or git repo -> incremental store update -> re-run
  the registered standing questions and report ranking diffs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vera.chassis.baseline import DenseChassis
from vera.chassis.preprocess import ChassisPreprocessor
from vera.stage2.fingerprint import FingerprintIndex
from vera.stage2.ingest import ingest_folder_snapshot, ingest_git_repo
from vera.stage2.ranker import RankedVersion, VersionRanker
from vera.stage2.store import VersionStore
from vera.verify.boost import TopKVerifier
from vera.verify.executor import VerificationSandbox
from vera.verify.parser import WorkedExampleParser


@dataclass
class Hit:
    doc_id: str
    code: str
    dense_rank: int
    final_rank: int
    dense_score: float
    final_score: float
    passed: int
    total: int
    confidence: float
    badges: List[str] = field(default_factory=list)


class DemoBackend:
    def __init__(
        self,
        corpus: Dict[str, str],
        encoder: str = "st",
        model_name: str = "Alibaba-NLP/gte-modernbert-base",
        verify_top_k: int = 30,
        alpha: float = 0.2,
        workers: Optional[int] = None,
        store_dir: Optional[Path] = None,
    ):
        self.raw_corpus = dict(corpus)
        self.pre = ChassisPreprocessor()
        self.chassis = DenseChassis(model_name=model_name, backend=encoder, batch_size=8, show_progress=False)
        self.chassis.index_corpus(self.pre.process_corpus(self.raw_corpus))
        self.sandbox = VerificationSandbox(default_timeout=0.75, reduced_timeout=0.3, workers=workers)
        self.verifier = TopKVerifier(sandbox=self.sandbox, default_alpha=alpha)
        self.parser = WorkedExampleParser()
        self.verify_top_k = verify_top_k
        self.alpha = alpha
        self.store = VersionStore(store_dir)
        self.fingerprints = FingerprintIndex(sandbox=self.sandbox)
        self.ranker = VersionRanker(self._encode, sandbox=self.sandbox, fingerprints=self.fingerprints)
        self.standing: Dict[str, str] = {}                  # name -> query text
        self.standing_last: Dict[str, List[str]] = {}       # name -> last ranked version ids (top of each chain)

    # ------------------------------------------------------------------ #
    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        emb = self.chassis.encode(list(texts))
        return emb.toarray().astype(np.float32) if hasattr(emb, "toarray") else emb

    def close(self) -> None:
        self.sandbox.close()

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, top_n: int = 10) -> Tuple[List[Hit], Dict[str, Any]]:
        t0 = time.time()
        clean = self.pre.process_queries({"q": query})
        dense = self.chassis.search(clean, top_k=max(self.verify_top_k, top_n))["q"]
        dense_rank = {d: i + 1 for i, d in enumerate(dense)}
        blended, qv = self.verifier.rerank_query(query, dense, self.raw_corpus, top_k=self.verify_top_k,
                                                 alpha=self.alpha, return_verification=True)
        hits: List[Hit] = []
        for r, (doc_id, score) in enumerate(list(blended.items())[:top_n], start=1):
            passed, total = qv.passed.get(doc_id, 0), qv.n_examples
            conf = qv.conf.get(doc_id, 0.0)
            badges = []
            if total:
                badges.append(f"PASSED {passed}/{total} examples" if passed == total else f"FAILED {passed}/{total} examples")
                if conf > 0:
                    badges.append(f"confidence {conf:.2f} (m={qv.m_first})")
            else:
                badges.append("no worked example — dense only")
            if dense_rank[doc_id] != r:
                badges.append(f"dense #{dense_rank[doc_id]} → #{r}")
            hits.append(Hit(doc_id, self.raw_corpus[doc_id], dense_rank[doc_id], r, dense[doc_id], score, passed, total, conf, badges))
        info = {"examples_parsed": qv.n_examples, "candidates_verified": len(qv.passed), "full_passers": qv.n_all_pass,
                "seconds": round(time.time() - t0, 2)}
        return hits, info

    # ------------------------------------------------------------------ #
    def ingest(self, source: str) -> Dict[str, Any]:
        """Folder snapshot or git repository -> store; then re-run standing questions."""
        src = Path(source)
        if (src / ".git").exists():
            reports = ingest_git_repo(self.store, src)
        else:
            reports = [ingest_folder_snapshot(self.store, src)]
        embedded = self.store.embed_missing(self._encode)
        diffs = self.rerun_standing()
        return {"snapshots": len(reports), "new_snippets": sum(r.new_snippets for r in reports),
                "unchanged_snippets": sum(r.unchanged_snippets for r in reports), "embedded": embedded,
                "store": self.store.stats(), "standing_diffs": diffs}

    def version_chain(self, path: str) -> List[Tuple[str, str, str]]:
        """[(version_label, ref, code)] for a stored path."""
        return [(f"v{i + 1}", v.ref, self.store.code(v.sid)) for i, v in enumerate(self.store.history(path))]

    def rank_chain(self, query: str, path: str) -> List[RankedVersion]:
        chain = self.version_chain(path)
        return self.ranker.rank(query, [(label, code) for label, _, code in chain])

    # ------------------------------------------------------------------ #
    def register_standing(self, name: str, query: str) -> None:
        self.standing[name] = query

    def rerun_standing(self) -> Dict[str, Dict[str, Any]]:
        """For each standing question: rank the latest version of every stored path; report what changed."""
        out: Dict[str, Dict[str, Any]] = {}
        latest = self.store.latest_ids()
        if not latest:
            return out
        ids = list(latest.values())
        for name, query in self.standing.items():
            q_emb = self._encode([query])[0]
            _, mat = self.store.matrix()
            sid_order, _ = self.store.matrix()
            pos = {s: i for i, s in enumerate(sid_order)}
            rows = [(p, s, float(q_emb @ mat[pos[s]] / (np.linalg.norm(q_emb) * np.linalg.norm(mat[pos[s]]) + 1e-9)))
                    for p, s in latest.items() if s in pos]
            rows.sort(key=lambda r: -r[2])
            top_paths = [p for p, _, _ in rows[:5]]
            prev = self.standing_last.get(name)
            out[name] = {"top_paths": top_paths, "changed": prev is not None and prev != top_paths, "previous": prev}
            self.standing_last[name] = top_paths
        return out


def load_demo_corpus(limit: Optional[int] = None) -> Tuple[Dict[str, str], Dict[str, str]]:
    """(corpus, example_queries) from the AppsRetrieval cache; ``limit`` shrinks the corpus for quick demos."""
    from vera.data.loader import AppsRetrievalDataset

    ds = AppsRetrievalDataset()
    corpus = ds.get_corpus()
    queries = ds.get_test_queries()
    if limit:
        keep = set(list(corpus)[:limit])
        # keep golds of the example queries inside the demo corpus
        qrels = ds.get_test_qrels()
        ex = dict(list(queries.items())[:8])
        for q in ex:
            keep |= set(qrels[q].keys())
        corpus = {d: corpus[d] for d in keep}
        queries = ex
    else:
        queries = dict(list(queries.items())[:8])
    return corpus, queries
