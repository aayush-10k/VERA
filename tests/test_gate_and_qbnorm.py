"""Tests for the R3 gate/router and the R4 QB-Norm hook."""

import numpy as np

from vera.chassis.qbnorm import QBNorm
from vera.gate.router import SignatureGate, UncertaintyRouter, example_shape, signature_compatible
from vera.gate.signature import extract_program_signature
from vera.verify.boost import GatedVerifier, QueryVerification, TopKVerifier, compute_rarity_confidence


def test_example_shape_layouts():
    s = example_shape("3\n1 2\n3 4\n5 6\n")
    assert s.n_lines == 4 and s.first_token_int == 3 and s.n_then_n_lines and not s.multi_case_t
    t = example_shape("2\n1 2\n3\n4 5 6\n7\n")
    assert t.multi_case_t and not t.n_then_n_lines
    assert example_shape("hello world\n").n_lines == 1 and example_shape("").n_lines == 0


def test_signature_gate_filters_incompatible_programs():
    corpus = {
        "single": "print(int(input()) * 2)",
        "ntn": "n=int(input())\nfor _ in range(n):\n    a,b=map(int,input().split())\n    print(a+b)",
        "func": "class Solution:\n    def f(self, x):\n        return x",
        "bad": "def f(:\n  pass",
    }
    gate = SignatureGate(corpus)
    assert set(gate.eligible) == {"single", "ntn"}
    assert gate.candidates("5\n") == ["single"]
    assert "ntn" in gate.candidates("3\n1 2\n3 4\n5 6\n") and "single" not in gate.candidates("3\n1 2\n3 4\n5 6\n")
    assert gate.coverage()["syntax_error"] == 1
    assert not signature_compatible(extract_program_signature(corpus["func"]), example_shape("5\n"))


def test_uncertainty_router_margin():
    r = UncertaintyRouter(tau=0.05)
    assert r.needs_corpus_wide({"a": 0.90, "b": 0.88, "c": 0.1})
    assert not r.needs_corpus_wide({"a": 0.90, "b": 0.70})
    assert not r.needs_corpus_wide({"a": 0.9})  # single candidate: nothing to be uncertain about


def test_gated_verifier_extends_only_uncertain_queries():
    gv = GatedVerifier(base_k=2, extend_k=5, tau=0.05, gate=None, workers=0)
    ordered_certain = [("a", 0.9), ("b", 0.5), ("c", 0.4), ("d", 0.3), ("e", 0.2), ("f", 0.1)]
    ordered_uncertain = [("a", 0.9), ("b", 0.89), ("c", 0.4), ("d", 0.3), ("e", 0.2), ("f", 0.1)]
    assert gv.candidate_ids("q", ordered_certain) == ["a", "b"]
    assert gv.candidate_ids("q", ordered_uncertain) == ["a", "b", "c", "d", "e"]
    assert gv.stats == {"queries": 2, "extended": 1, "extra_candidates": 3}
    gv.close()


def test_blend_is_bounded_and_monotone():
    qv = QueryVerification(n_examples=2)
    qv.conf = {"gold": compute_rarity_confidence(2, 2, 1), "impostor": compute_rarity_confidence(2, 2, 64)}
    dense = {"top": 1.0, "gold": 0.5, "impostor": 0.5, "tail": 0.0}
    out = TopKVerifier.blend(dense, qv, alpha=0.3, top_k=3)
    assert out["gold"] > out["impostor"] > out["tail"]
    assert abs(out["gold"] - (0.5 + 0.3 * 1.0)) < 1e-9
    assert out["top"] == 1.0  # unverified/unboosted top keeps its normalized dense score
    assert out["impostor"] < 0.5 + 0.3 * 0.15  # rarity discount: 1/(1+log2 64) = 1/7


def test_qbnorm_demotes_bank_members_only():
    rng = np.random.default_rng(0)
    docs = rng.normal(size=(6, 16)).astype(np.float32)
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    bank = np.vstack([docs[:2] + 0.01, rng.normal(size=(10, 16))]).astype(np.float32)
    bank /= np.linalg.norm(bank, axis=1, keepdims=True)
    qb = QBNorm.fit_hub(docs, bank, k=1, beta=0.5)
    assert qb.hub[:2].min() > qb.hub[2:].max()
    sims = np.ones((1, 6), dtype=np.float32)
    out = qb(["q"], sims)
    assert out[0, :2].max() < out[0, 2:].min()
    assert np.array_equal(QBNorm(hub=qb.hub, beta=0.0)(["q"], sims), sims)
