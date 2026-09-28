"""
Stage-2 multi-source ingestion (S1)
===================================
Two version sources feed the same :class:`VersionStore`:

* **git repositories** — every commit (oldest first) becomes a snapshot; only the ``*.py``
  blobs changed by that commit are read (``git diff-tree``), the rest are carried over.
* **folder snapshots** — a directory tree scanned at a point in time; successive scans of
  the same tree (or of copies ``v1/``, ``v2/``, ...) become successive snapshots.

Both call ``store.ingest(records, snapshot=...)`` and return the per-snapshot reports, from
which the rebuild benchmark (full re-embed vs incremental) is computed.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from vera.stage2.store import IngestReport, SnippetRecord, VersionStore

PY_SUFFIXES = (".py",)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
                          errors="replace").stdout


def _is_code(path: str, suffixes: Sequence[str]) -> bool:
    return path.endswith(tuple(suffixes)) and "/." not in "/" + path


# --------------------------------------------------------------------------- #
def ingest_folder_snapshot(store: VersionStore, folder: Path, label: Optional[str] = None,
                           suffixes: Sequence[str] = PY_SUFFIXES) -> IngestReport:
    """Scan ``folder`` recursively and register it as one snapshot."""
    folder = Path(folder)
    label = label or f"folder:{folder.name}@{int(time.time())}"
    records: List[SnippetRecord] = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules")]
        for fn in sorted(files):
            if not fn.endswith(tuple(suffixes)):
                continue
            p = Path(root) / fn
            try:
                code = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            records.append(SnippetRecord(path=str(p.relative_to(folder)), code=code, source="folder", ref=label,
                                         timestamp=p.stat().st_mtime))
    return store.ingest(records, snapshot=label)


def ingest_folder_series(store: VersionStore, folders: Iterable[Path]) -> List[IngestReport]:
    """Ingest several folder snapshots in order (e.g. ``v1/``, ``v2/``, ``v3/``)."""
    return [ingest_folder_snapshot(store, Path(f), label=f"folder:{Path(f).name}") for f in folders]


# --------------------------------------------------------------------------- #
def list_commits(repo: Path, branch: Optional[str] = None, max_commits: Optional[int] = None) -> List[Dict[str, object]]:
    """Commits oldest-first: [{sha, time}]."""
    args = ["log", "--reverse", "--format=%H %ct"]
    if branch:
        args.append(branch)
    out = _git(repo, *args).strip().splitlines()
    commits = [{"sha": ln.split()[0], "time": float(ln.split()[1])} for ln in out if ln.strip()]
    return commits[-max_commits:] if max_commits else commits


def ingest_git_repo(store: VersionStore, repo: Path, branch: Optional[str] = None, max_commits: Optional[int] = None,
                    suffixes: Sequence[str] = PY_SUFFIXES) -> List[IngestReport]:
    """Replay a repository's history; each commit is one snapshot. Only changed blobs are read from git."""
    repo = Path(repo)
    commits = list_commits(repo, branch=branch, max_commits=max_commits)
    reports: List[IngestReport] = []
    current: Dict[str, str] = {}   # path -> code at the current commit
    prev_sha: Optional[str] = None
    for c in commits:
        sha = str(c["sha"])
        if prev_sha is None:
            listing = _git(repo, "ls-tree", "-r", "--name-only", sha).splitlines()
            changed = [p for p in listing if _is_code(p, suffixes)]
            removed: List[str] = []
        else:
            diff = _git(repo, "diff-tree", "-r", "--name-status", "--no-renames", prev_sha, sha).splitlines()
            changed, removed = [], []
            for ln in diff:
                if not ln.strip():
                    continue
                status, path = ln.split("\t", 1)
                if not _is_code(path, suffixes):
                    continue
                (removed if status.startswith("D") else changed).append(path)
        for path in removed:
            current.pop(path, None)
        for path in changed:
            try:
                current[path] = _git(repo, "show", f"{sha}:{path}")
            except subprocess.CalledProcessError:
                current.pop(path, None)
        records = [SnippetRecord(path=p, code=code, source="git", ref=sha, timestamp=float(c["time"]))
                   for p, code in sorted(current.items())]
        reports.append(store.ingest(records, snapshot=f"git:{sha[:10]}"))
        prev_sha = sha
    return reports


# --------------------------------------------------------------------------- #
def rebuild_benchmark(reports: Sequence[IngestReport], per_snippet_embed_s: float) -> Dict[str, float]:
    """Full re-embed cost vs incremental cost across the ingested snapshots (embedding time modelled per snippet)."""
    full = sum((r.new_snippets + r.unchanged_snippets) * per_snippet_embed_s for r in reports)
    incremental = sum(r.new_snippets * per_snippet_embed_s for r in reports)
    return {"snapshots": len(reports), "full_reembed_s": round(full, 3), "incremental_s": round(incremental, 3),
            "speedup": round(full / incremental, 2) if incremental > 0 else float("inf")}
