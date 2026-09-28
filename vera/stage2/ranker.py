"""
Stage-2 version ranking (S3)
============================
Given several versions of a program that answer the same query, rank them so the
*working* version comes first:

1. **Execution separation** — versions are first ordered by how many worked examples they
   pass (a version that fails the sample never outranks one that passes it).
2. **Probe consensus** (differential testing) — for versions that tie on the sample, each
   version's behaviour fingerprint (8-probe battery mutated from the worked example) is
   compared with every other version's; the version whose behaviour agrees with most of
   its peers wins the tie. A one-token bug typically changes the output on some probe while
   the fix and its refactors agree with each other.
3. **Diff-line ranker** — unified diff between consecutive versions; only the added/changed
   lines are embedded and blended with the global similarity ``0.7 * S_global + 0.3 * S_diff``
   (the first version has no diff, so its blend is its global score).
4. **Trace-value tie-break** (narrow tier) — when the statement carries ≥ 3 distinctive
   literal numbers, the int/str locals observed while running the sample (``sys.settrace``
   in the sandbox child) are compared with the statement's literals (Jaccard).

Fingerprints also certify *behaviorally unchanged* revisions, which inherit the verification
result of their predecessor instead of being re-run.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vera.stage2.fingerprint import Fingerprint, FingerprintIndex, fingerprint_agreement
from vera.verify.executor import VerificationSandbox
from vera.verify.parser import WorkedExampleParser, statement_allows_any_order

EncodeFn = Callable[[Sequence[str]], np.ndarray]

_NUM_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")


# --------------------------------------------------------------------------- #
def diff_lines(old_code: str, new_code: str) -> Tuple[List[str], List[str]]:
    """(added_or_changed, removed) source lines between two versions (no diff headers)."""
    added, removed = [], []
    for ln in difflib.unified_diff(old_code.splitlines(), new_code.splitlines(), lineterm="", n=0):
        if ln.startswith(("+++", "---", "@@")):
            continue
        if ln.startswith("+") and ln[1:].strip():
            added.append(ln[1:])
        elif ln.startswith("-") and ln[1:].strip():
            removed.append(ln[1:])
    return added, removed


def statement_numbers(statement: str, min_count: int = 3) -> List[str]:
    """Distinctive numeric literals in a statement (excluding tiny constants); empty if fewer than ``min_count``."""
    nums = {m.group(0) for m in _NUM_RE.finditer(statement)}
    nums = {n for n in nums if n.lstrip("-") not in ("0", "1", "2", "10", "100")}
    return sorted(nums) if len(nums) >= min_count else []


def trace_overlap(trace_values: Optional[Sequence[str]], numbers: Sequence[str]) -> float:
    """Jaccard overlap between statement literals and the values observed at runtime."""
    if not trace_values or not numbers:
        return 0.0
    a, b = set(numbers), set(trace_values)
    return len(a & b) / len(a | b)


# --------------------------------------------------------------------------- #
@dataclass
class RankedVersion:
    version_id: str
    rank: int
    passed: int
    total: int
    global_score: float
    diff_score: float
    blended: float
    consensus: float
    trace_jaccard: float
    fingerprint: Optional[str]
    badges: List[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


class VersionRanker:
    def __init__(
        self,
        encode_fn: EncodeFn,
        sandbox: Optional[VerificationSandbox] = None,
        w_global: float = 1.0,
        w_diff: float = 0.0,
        parser: Optional[WorkedExampleParser] = None,
        fingerprints: Optional[FingerprintIndex] = None,
        use_consensus: bool = True,
    ):
        # Defaults measured on docs/stage2-benchmark.md: the diff-line term (plan default 0.7/0.3) LOSES to plain
        # global similarity once probe consensus is in place (both-pass working-first 72.9% -> 86.4% with the gte
        # encoder), so it is off by default and kept only as an ablation (scripts/s3_version_benchmark.py --w-diff 0.3).
        self.encode_fn = encode_fn
        self.sandbox = sandbox or VerificationSandbox(default_timeout=0.75, reduced_timeout=0.3)
        self.w_global = w_global
        self.w_diff = w_diff
        self.parser = parser or WorkedExampleParser()
        self.fingerprints = fingerprints or FingerprintIndex(sandbox=self.sandbox)
        self.use_consensus = use_consensus

    def close(self) -> None:
        self.sandbox.close()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0

    def rank(
        self,
        query_text: str,
        versions: Sequence[Tuple[str, str]],
        query_embedding: Optional[np.ndarray] = None,
        use_fingerprints: bool = True,
        peer_fingerprints: Sequence[Fingerprint] = (),
    ) -> List[RankedVersion]:
        """Rank ``[(version_id, code), ...]`` (chronological order) for one query.

        ``peer_fingerprints``: fingerprints of *other* programs known to pass this query's sample (e.g. other
        corpus candidates); they join the consensus vote.
        """
        if not versions:
            return []
        examples = self.parser.parse_examples(query_text)
        any_order = statement_allows_any_order(query_text)
        numbers = statement_numbers(query_text)
        q_emb = query_embedding if query_embedding is not None else self.encode_fn([query_text])[0]

        codes = [c for _, c in versions]
        global_embs = self.encode_fn(codes)
        diff_texts: List[Optional[str]] = [None]
        for i in range(1, len(codes)):
            added, removed = diff_lines(codes[i - 1], codes[i])
            diff_texts.append("\n".join(added) if added else ("\n".join(removed) if removed else None))
        to_embed = [t for t in diff_texts if t]
        diff_emb_iter = iter(self.encode_fn(to_embed)) if to_embed else iter(())

        rows: List[RankedVersion] = []
        fps: List[Optional[Fingerprint]] = []
        prev_row: Optional[RankedVersion] = None
        for i, (vid, code) in enumerate(versions):
            badges: List[str] = []
            fp = None
            if use_fingerprints and examples:
                fp = self.fingerprints.fingerprint(vid, code, examples[0].stdin)
                badges.append(f"fingerprint {fp.short}")
            fps.append(fp)
            g = self._cos(q_emb, global_embs[i])
            d = self._cos(q_emb, next(diff_emb_iter)) if diff_texts[i] else g
            if fp is not None and prev_row is not None and fps[i - 1] is not None and fp.digest == fps[i - 1].digest:
                row = RankedVersion(vid, 0, prev_row.passed, prev_row.total, g, d, 0.0, 0.0, prev_row.trace_jaccard, fp.digest,
                                    badges + ["behaviorally unchanged"])
            else:
                passed, total, jacc = 0, len(examples), 0.0
                if examples:
                    res = self.sandbox.verify_candidate(code, examples, multiline_set=any_order)
                    passed = res.passed_examples
                    if numbers and res.results:
                        tr = self.sandbox.execute_snippet(code, examples[0].stdin, expected_output=examples[0].expected_stdout,
                                                          trace_values=True)
                        jacc = trace_overlap(tr.trace_values, numbers)
                    badges.append(f"PASSED {passed}/{total} examples" if total and passed == total else f"FAILED {passed}/{total} examples")
                row = RankedVersion(vid, 0, passed, total, g, d, 0.0, 0.0, jacc, fp.digest if fp else None, badges)
            row.blended = self.w_global * row.global_score + self.w_diff * row.diff_score
            rows.append(row)
            prev_row = row

        # probe consensus: agreement of each version's behaviour with the other versions (+ external peers)
        if self.use_consensus:
            for i, row in enumerate(rows):
                if fps[i] is None:
                    continue
                others = [f for j, f in enumerate(fps) if j != i and f is not None] + list(peer_fingerprints)
                row.consensus = sum(fingerprint_agreement(fps[i], f) for f in others) / len(others) if others else 0.0

        order = sorted(range(len(rows)), key=lambda k: (-round(rows[k].pass_rate, 6), -round(rows[k].consensus, 4),
                                                        -round(rows[k].blended, 4), -round(rows[k].trace_jaccard, 4), -k))
        for r, k in enumerate(order, start=1):
            rows[k].rank = r
        return sorted(rows, key=lambda r: r.rank)
