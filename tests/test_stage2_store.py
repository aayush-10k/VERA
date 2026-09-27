"""Tests for the Stage-2 version store and ingestion (S1)."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from vera.stage2.ingest import ingest_folder_series, ingest_git_repo, rebuild_benchmark
from vera.stage2.store import VersionStore, normalize_code, snippet_id


def test_snippet_id_ignores_formatting_comments_docstrings():
    a = "def f(x):\n    # comment\n    return x+1\n"
    b = "def f(x):\n    '''doc'''\n    return  x + 1   # other\n\n"
    c = "def f(x):\n    return x+2\n"
    assert snippet_id(a) == snippet_id(b)
    assert snippet_id(a) != snippet_id(c)


def test_unparseable_code_falls_back_to_text_hash():
    norm, method = normalize_code("print x  # py2\n")
    assert method == "text" and norm == "print x"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_git_ingest_is_incremental(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    for i in range(10):
        (repo / f"m{i}.py").write_text(f"def f{i}(x):\n    return x*{i}\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "v1")
    (repo / "m3.py").write_text("def f3(x):\n    return x*3 + 1\n")
    _git(repo, "commit", "-q", "-am", "v2")
    (repo / "m5.py").write_text("def f5(x):\n    # reformatted only\n    return  x * 5\n")
    (repo / "m6.py").unlink()
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "v3")

    store = VersionStore(tmp_path / "store")
    reports = ingest_git_repo(store, repo)
    assert [r.new_snippets for r in reports] == [10, 1, 0]
    assert [r.unchanged_snippets for r in reports] == [0, 9, 9]  # m5 reformat keeps its id, m6 deleted
    assert len(store.history("m3.py")) == 2
    assert len(store.history("m5.py")) == 1  # reformat keeps the id
    bench = rebuild_benchmark(reports, per_snippet_embed_s=1.0)
    assert bench["speedup"] > 2.0

    embedded = store.embed_missing(lambda codes: np.ones((len(codes), 4), dtype=np.float32))
    assert embedded == 11 and store.missing_embeddings() == []
    store.save()
    reloaded = VersionStore(tmp_path / "store")
    assert reloaded.stats() == store.stats()


def test_folder_series(tmp_path: Path):
    for v in ("v1", "v2"):
        d = tmp_path / v
        d.mkdir()
        (d / "a.py").write_text("print(1)\n")
        (d / "b.py").write_text(f"print({v!r})\n")
    reports = ingest_folder_series(VersionStore(), [tmp_path / "v1", tmp_path / "v2"])
    assert [(r.new_snippets, r.unchanged_snippets) for r in reports] == [(2, 0), (1, 1)]
