"""Headless tests for the demo backend (no dataset download, TF-IDF chassis)."""

from pathlib import Path

import pytest

from vera.demo.backend import DemoBackend

QUERY = """Given n, print n squared.

-----Sample Input-----
4
-----Sample Output-----
16
"""
CORPUS = {
    "sq": "n = int(input())\nprint(n * n)\n",
    "dbl": "n = int(input())\nprint(n * 2)\n",
    "rev": "s = input()\nprint(s[::-1])\n",
}


@pytest.fixture(scope="module")
def backend():
    be = DemoBackend(CORPUS, encoder="tfidf", verify_top_k=3, alpha=1.5, workers=1)
    yield be
    be.close()


def test_retrieve_badges_and_boost(backend):
    hits, info = backend.retrieve(QUERY, top_n=3)
    assert info["examples_parsed"] == 1 and info["candidates_verified"] == 3
    assert hits[0].doc_id == "sq" and "PASSED 1/1 examples" in hits[0].badges
    assert all("FAILED" in " ".join(h.badges) for h in hits[1:])


def test_ingest_versions_and_standing_questions(backend, tmp_path: Path):
    for v, code in (("v1", CORPUS["dbl"]), ("v2", CORPUS["sq"]), ("v3", CORPUS["sq"].replace("print", "print ")), ):
        d = tmp_path / v
        d.mkdir()
        (d / "solve.py").write_text(code)
    backend.register_standing("square", QUERY)
    reps = [backend.ingest(str(tmp_path / v)) for v in ("v1", "v2", "v3")]
    assert [r["new_snippets"] for r in reps] == [1, 1, 0]          # v3 is a reformat of v2
    assert reps[-1]["standing_diffs"]["square"]["top_paths"] == ["solve.py"]
    chain = backend.version_chain("solve.py")
    assert [c[0] for c in chain] == ["v1", "v2"]
    ranked = backend.rank_chain(QUERY, "solve.py")
    assert ranked[0].version_id == "v2" and ranked[0].passed == 1 and ranked[1].passed == 0
