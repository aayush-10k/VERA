"""
Stage-2 version store (S1)
==========================
Content-addressed store for code snippets across revisions.

* ``snippet_id(code)`` = SHA-256 of the *normalized AST dump* (comments, docstrings,
  whitespace, line numbers and formatting stripped). Moving or reformatting a function
  keeps its id; only a genuine AST change produces a new id. Code that does not parse
  (e.g. Python 2) falls back to a whitespace-normalized text hash.
* The store keeps every distinct snippet once, a per-path version chain across snapshots
  (git commits or folder snapshots), and an embedding cache keyed by snippet id, so a new
  revision only re-embeds the ids that were never seen before (incremental rebuild).

Persistence is a directory: ``index.json`` (snippets, chains, snapshots) + ``embeddings.npz``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Normalisation & hashing
# --------------------------------------------------------------------------- #
class _DocstringStripper(ast.NodeTransformer):
    def _strip(self, node):
        body = getattr(node, "body", None)
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
        return node

    def visit_Module(self, node):
        self.generic_visit(node)
        return self._strip(node)

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        return self._strip(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        return self._strip(node)


def normalize_code(code: str) -> Tuple[str, str]:
    """Return (normalized_text, method). method is 'ast' or 'text' (unparseable code)."""
    src = code.replace("\r\n", "\n").replace("\r", "\n")
    try:
        tree = ast.parse(src)
        tree = _DocstringStripper().visit(tree)
        return ast.dump(tree, annotate_fields=True, include_attributes=False), "ast"
    except SyntaxError:
        text = "\n".join(re.sub(r"\s+", " ", ln.split("#", 1)[0]).strip() for ln in src.split("\n"))
        text = re.sub(r"\n+", "\n", text).strip()
        return text, "text"


def snippet_id(code: str) -> str:
    norm, method = normalize_code(code)
    return hashlib.sha256((method + "\x00" + norm).encode("utf-8", errors="ignore")).hexdigest()


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class SnippetRecord:
    path: str                 # logical location (file path inside the repo / folder)
    code: str
    source: str               # 'git' | 'folder'
    ref: str                  # commit sha or snapshot label
    timestamp: float = 0.0    # commit time or mtime


@dataclass
class VersionEntry:
    sid: str
    ref: str
    source: str
    timestamp: float
    snapshot_index: int


@dataclass
class IngestReport:
    snapshot: str
    records: int
    new_snippets: int
    unchanged_snippets: int
    new_ids: List[str] = field(default_factory=list)
    elapsed_s: float = 0.0


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
class VersionStore:
    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else None
        self.snippets: Dict[str, Dict[str, object]] = {}      # sid -> {code, first_ref, first_seen}
        self.chains: Dict[str, List[VersionEntry]] = {}       # path -> versions in ingestion order
        self.snapshots: List[Dict[str, object]] = []          # [{ref, source, n_records, n_new}]
        self.embeddings: Dict[str, np.ndarray] = {}           # sid -> vector
        if self.root and (self.root / "index.json").exists():
            self.load()

    # ----- persistence ------------------------------------------------------ #
    def save(self) -> None:
        if not self.root:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        idx = {
            "snippets": self.snippets,
            "chains": {p: [asdict(v) for v in vs] for p, vs in self.chains.items()},
            "snapshots": self.snapshots,
        }
        (self.root / "index.json").write_text(json.dumps(idx))
        if self.embeddings:
            np.savez_compressed(self.root / "embeddings.npz", ids=np.array(list(self.embeddings.keys())),
                                vecs=np.stack(list(self.embeddings.values())))

    def load(self) -> None:
        assert self.root is not None
        idx = json.loads((self.root / "index.json").read_text())
        self.snippets = idx["snippets"]
        self.chains = {p: [VersionEntry(**v) for v in vs] for p, vs in idx["chains"].items()}
        self.snapshots = idx["snapshots"]
        emb = self.root / "embeddings.npz"
        if emb.exists():
            data = np.load(emb, allow_pickle=False)
            self.embeddings = dict(zip(data["ids"].tolist(), data["vecs"]))

    # ----- ingestion -------------------------------------------------------- #
    def ingest(self, records: Iterable[SnippetRecord], snapshot: str) -> IngestReport:
        """Register one snapshot (all files of one commit / one folder state). Returns the new-vs-unchanged split."""
        t0 = time.perf_counter()
        recs = list(records)
        snap_index = len(self.snapshots)
        new_ids: List[str] = []
        unchanged = 0
        for r in recs:
            sid = snippet_id(r.code)
            if sid not in self.snippets:
                self.snippets[sid] = {"code": r.code, "first_ref": r.ref, "first_path": r.path, "first_seen": r.timestamp}
                new_ids.append(sid)
            else:
                unchanged += 1
            chain = self.chains.setdefault(r.path, [])
            if not chain or chain[-1].sid != sid:
                chain.append(VersionEntry(sid=sid, ref=r.ref, source=r.source, timestamp=r.timestamp, snapshot_index=snap_index))
        self.snapshots.append({"ref": snapshot, "source": recs[0].source if recs else "", "n_records": len(recs), "n_new": len(new_ids)})
        return IngestReport(snapshot=snapshot, records=len(recs), new_snippets=len(new_ids), unchanged_snippets=unchanged,
                            new_ids=new_ids, elapsed_s=time.perf_counter() - t0)

    # ----- embeddings ------------------------------------------------------- #
    def missing_embeddings(self) -> List[str]:
        return [sid for sid in self.snippets if sid not in self.embeddings]

    def embed_missing(self, encode_fn: Callable[[Sequence[str]], np.ndarray], batch: int = 64) -> int:
        """Embed only snippets without a cached vector. Returns how many were embedded."""
        missing = self.missing_embeddings()
        for start in range(0, len(missing), batch):
            ids = missing[start:start + batch]
            vecs = encode_fn([str(self.snippets[s]["code"]) for s in ids])
            for sid, v in zip(ids, np.asarray(vecs)):
                self.embeddings[sid] = np.asarray(v, dtype=np.float32)
        return len(missing)

    def matrix(self) -> Tuple[List[str], np.ndarray]:
        ids = [s for s in self.snippets if s in self.embeddings]
        if not ids:
            return [], np.zeros((0, 0), dtype=np.float32)
        return ids, np.stack([self.embeddings[s] for s in ids])

    # ----- queries ---------------------------------------------------------- #
    def history(self, path: str) -> List[VersionEntry]:
        return list(self.chains.get(path, []))

    def latest_ids(self) -> Dict[str, str]:
        """path -> sid of the most recent version."""
        return {p: vs[-1].sid for p, vs in self.chains.items() if vs}

    def code(self, sid: str) -> str:
        return str(self.snippets[sid]["code"])

    def paths_of(self, sid: str) -> List[str]:
        return [p for p, vs in self.chains.items() if any(v.sid == sid for v in vs)]

    def stats(self) -> Dict[str, int]:
        return {"snippets": len(self.snippets), "paths": len(self.chains), "snapshots": len(self.snapshots),
                "embedded": len(self.embeddings), "versions": sum(len(v) for v in self.chains.values())}
