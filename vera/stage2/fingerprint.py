"""
Stage-2 behaviour fingerprints (S2)
===================================
Two versions of a program are "the same" when they *do* the same thing, not when they
*read* the same. A fingerprint is the SHA-256 of a program's normalized outputs over a
deterministic **probe battery**; identical fingerprints across versions certify the change
as *behaviorally unchanged* (so re-ranking / re-verification can be skipped).

Probe battery
-------------
Probes are derived from the worked example of the query the version chain answers, so they
respect the real input grammar: probe 0 is the example itself; probes 1..n-1 mutate the
example's numeric tokens (same layout, values of similar magnitude, seeded RNG) and, for
``n-then-n-lines`` / ``multi-case-t`` layouts, keep the count fields consistent. Crashes and
timeouts are fingerprinted too (``CRASH:<ExcType>`` / ``TIMEOUT``), because a version that
crashes where the previous one did not is a behaviour change.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from vera.gate.router import example_shape
from vera.stage2.store import snippet_id
from vera.verify.executor import VerificationSandbox

_INT_RE = re.compile(r"^[-+]?\d+$")
_FLOAT_RE = re.compile(r"^[-+]?\d+\.\d+$")


def _mutate_int(tok: str, rng: random.Random) -> str:
    v = int(tok)
    mag = max(1, abs(v))
    lo, hi = (1, max(2, 2 * mag)) if v > 0 else (-2 * mag, 2 * mag)
    return str(rng.randint(lo, hi))


def _mutate_line(line: str, rng: random.Random, keep_count: bool) -> str:
    toks = line.split()
    out = []
    for t in toks:
        if _INT_RE.match(t) and not keep_count:
            out.append(_mutate_int(t, rng))
        elif _FLOAT_RE.match(t):
            out.append(f"{float(t) * rng.uniform(0.5, 1.5):.3f}")
        elif t.isalpha() and len(t) > 1:
            letters = sorted(set(t.lower())) or ["a"]
            out.append("".join(rng.choice(letters) for _ in range(len(t))))
        else:
            out.append(t)
    return " ".join(out)


def build_probe_battery(example_stdin: str, n_probes: int = 8, seed: int = 1234) -> List[str]:
    """Probe 0 is the example itself; the rest are layout-preserving numeric/string mutations."""
    rng = random.Random(seed)
    base_lines = [ln for ln in example_stdin.replace("\r", "").split("\n") if ln.strip()]
    if not base_lines:
        return [example_stdin] + [f"{rng.randint(1, 100)}\n" for _ in range(n_probes - 1)]
    shape = example_shape(example_stdin)
    probes = [example_stdin if example_stdin.endswith("\n") else example_stdin + "\n"]
    for _ in range(n_probes - 1):
        lines = []
        for i, ln in enumerate(base_lines):
            # keep count fields (first line of n-then-n / multi-case layouts) so the layout stays valid
            keep = i == 0 and (shape.n_then_n_lines or shape.multi_case_t) and shape.first_line_tokens == 1
            lines.append(_mutate_line(ln, rng, keep_count=keep))
        probes.append("\n".join(lines) + "\n")
    return probes


@dataclass
class Fingerprint:
    digest: str
    outputs: List[str] = field(default_factory=list)   # normalized per-probe output tokens
    n_probes: int = 0
    n_crashes: int = 0
    n_timeouts: int = 0

    @property
    def short(self) -> str:
        return "#" + self.digest[:8]


def _normalize_output(res) -> str:
    if res.status == "timeout":
        return "TIMEOUT"
    if res.status in ("error", "syntax_error"):
        exc = (res.error_message or "error").split(":")[0].strip()
        return f"CRASH:{exc}"
    return " ".join(res.stdout.split())


def fingerprint_program(code: str, probes: Sequence[str], sandbox: VerificationSandbox) -> Fingerprint:
    outputs: List[str] = []
    crashes = timeouts = 0
    for p in probes:
        res = sandbox.execute_snippet(code, p)
        norm = _normalize_output(res)
        outputs.append(norm)
        crashes += norm.startswith("CRASH:")
        timeouts += norm == "TIMEOUT"
    digest = hashlib.sha256("\x1e".join(outputs).encode("utf-8", errors="ignore")).hexdigest()
    return Fingerprint(digest=digest, outputs=outputs, n_probes=len(probes), n_crashes=crashes, n_timeouts=timeouts)


def behaviorally_unchanged(a: Fingerprint, b: Fingerprint) -> bool:
    return a.digest == b.digest


def fingerprint_agreement(a: Fingerprint, b: Fingerprint) -> float:
    """Fraction of probes on which two programs agree (1.0 == identical fingerprint)."""
    if not a.outputs or len(a.outputs) != len(b.outputs):
        return 0.0
    return sum(x == y for x, y in zip(a.outputs, b.outputs)) / len(a.outputs)


class FingerprintIndex:
    """snippet id -> Fingerprint, with a per-query probe battery cache."""

    def __init__(self, sandbox: Optional[VerificationSandbox] = None, n_probes: int = 8):
        self.sandbox = sandbox or VerificationSandbox(default_timeout=0.75, reduced_timeout=0.3)
        self.n_probes = n_probes
        self.fingerprints: Dict[Tuple[str, str], Fingerprint] = {}   # (sid, battery_key) -> fp
        self.batteries: Dict[str, List[str]] = {}

    def battery_for(self, example_stdin: str) -> Tuple[str, List[str]]:
        key = hashlib.sha1(example_stdin.encode("utf-8", errors="ignore")).hexdigest()[:12]
        if key not in self.batteries:
            self.batteries[key] = build_probe_battery(example_stdin, n_probes=self.n_probes)
        return key, self.batteries[key]

    def fingerprint(self, sid: str, code: str, example_stdin: str) -> Fingerprint:
        """``sid`` is only a label; the cache is keyed by the code's normalized-AST id, so a reformatted version
        shares its predecessor's fingerprint without being re-run (identical AST => identical behaviour)."""
        key, probes = self.battery_for(example_stdin)
        cache_key = (snippet_id(code), key)
        if cache_key not in self.fingerprints:
            self.fingerprints[cache_key] = fingerprint_program(code, probes, self.sandbox)
        return self.fingerprints[cache_key]

    def close(self) -> None:
        self.sandbox.close()
