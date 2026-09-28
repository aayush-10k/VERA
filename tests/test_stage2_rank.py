"""Tests for Stage-2 fingerprints and version ranking (S2/S3)."""

import numpy as np
import pytest

from vera.stage2.fingerprint import FingerprintIndex, build_probe_battery, fingerprint_agreement, fingerprint_program
from vera.stage2.ranker import VersionRanker, diff_lines, statement_numbers, trace_overlap
from vera.verify.executor import VerificationSandbox

QUERY = """Print the sum of each of the n pairs.

-----Sample Input-----
3
1 2
3 4
5 6
-----Sample Output-----
3
7
11
"""
V_OK = "n=int(input())\nfor _ in range(n):\n    a,b=map(int,input().split())\n    print(a+b)\n"
V_REF = "n = int(input())  # pairs\nfor i in range(n):\n    x, y = map(int, input().split())\n    print(x + y)\n"
V_BUG = "n=int(input())\nfor _ in range(n):\n    a,b=map(int,input().split())\n    print(a+b if a<7 else a-b)\n"  # passes the sample


def _tfidf_encoder(texts_for_fit):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(token_pattern=r"\S+").fit(texts_for_fit)
    return lambda texts: vec.transform(list(texts)).toarray().astype(np.float32)


@pytest.fixture(scope="module")
def sandbox():
    sb = VerificationSandbox(default_timeout=0.75, reduced_timeout=0.3, workers=2)
    yield sb
    sb.close()


def test_probe_battery_keeps_layout():
    probes = build_probe_battery("3\n1 2\n3 4\n5 6\n", n_probes=4)
    assert probes[0] == "3\n1 2\n3 4\n5 6\n"
    for p in probes[1:]:
        lines = p.strip().split("\n")
        assert lines[0] == "3" and len(lines) == 4 and all(len(ln.split()) == 2 for ln in lines[1:])


def test_fingerprint_certifies_refactor_and_exposes_hidden_bug(sandbox):
    probes = build_probe_battery("3\n1 2\n3 4\n5 6\n", n_probes=6)
    f_ok, f_ref, f_bug = (fingerprint_program(c, probes, sandbox) for c in (V_OK, V_REF, V_BUG))
    assert f_ok.digest == f_ref.digest
    assert f_ok.outputs[0] == f_bug.outputs[0]          # identical on the worked example ...
    assert f_ok.digest != f_bug.digest                  # ... but the probe battery separates them
    assert 0.0 < fingerprint_agreement(f_ok, f_bug) < 1.0


def test_ranker_puts_working_version_first_in_both_pass_case(sandbox):
    enc = _tfidf_encoder([QUERY, V_OK, V_REF, V_BUG])
    ranker = VersionRanker(enc, sandbox=sandbox, fingerprints=FingerprintIndex(sandbox=sandbox, n_probes=6))
    for chain in ([("v1", V_BUG), ("v2", V_OK), ("v3", V_REF)], [("v1", V_OK), ("v2", V_REF), ("v3", V_BUG)]):
        ranked = ranker.rank(QUERY, chain)
        assert ranked[0].version_id != next(v for v, c in chain if c == V_BUG)
        by_id = {r.version_id: r for r in ranked}
        assert all(r.passed == r.total == 1 for r in ranked)      # everything passes the sample
        ref_id = next(v for v, c in chain if c == V_REF)
        assert "behaviorally unchanged" in by_id[ref_id].badges


def test_helpers():
    added, removed = diff_lines("a = 1\nb = 2\n", "a = 1\nb = 3\n")
    assert added == ["b = 3"] and removed == ["b = 2"]
    assert statement_numbers("n up to 1000, answer modulo 998244353, at most 45 steps") == ["1000", "45", "998244353"]
    assert statement_numbers("just 1 and 2") == []
    assert trace_overlap(["45", "7"], ["45", "998244353"]) == pytest.approx(1 / 3)
