"""
Corpus-wide gate & uncertainty router (Rung R3)
==============================================
Active only because the M4 gate passed (docs/goldrun.md). Two pieces:

1. **Uncertainty router** — if the dense top-1/top-2 margin is at least ``tau`` the dense
   answer is treated as decisive and the query gets no corpus-wide verification (top-K
   verification still applies). ``tau`` is fit on dev.
2. **Signature layout gate** — for the remaining queries, only corpus programs whose static
   I/O signature is compatible with the shape of the query's worked example are eligible
   for corpus-wide execution. ``unknown`` and ``functional`` programs are excluded (they
   are handled by the top-K path only).

Both are pure functions of text and static analysis; no dataset metadata is read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from vera.gate.signature import ProgramSignature, extract_program_signature


@dataclass(frozen=True)
class ExampleShape:
    n_lines: int
    first_line_tokens: int
    first_token_int: Optional[int]
    all_numeric: bool
    n_then_n_lines: bool      # first line is a single int n and exactly n further lines follow
    multi_case_t: bool        # first line is a single int t and the remaining lines split into t blocks


def example_shape(stdin_text: str) -> ExampleShape:
    lines = [ln for ln in stdin_text.replace("\r", "").split("\n") if ln.strip()]
    if not lines:
        return ExampleShape(0, 0, None, False, False, False)
    first_tokens = lines[0].split()
    first_int: Optional[int] = None
    if len(first_tokens) == 1 and re.fullmatch(r"[-+]?\d+", first_tokens[0]):
        first_int = int(first_tokens[0])
    all_numeric = all(re.fullmatch(r"[-+]?\d+(\.\d+)?", tok) for ln in lines for tok in ln.split())
    rest = len(lines) - 1
    n_then_n = first_int is not None and first_int > 0 and rest == first_int
    multi_t = first_int is not None and 1 <= first_int < rest and rest % first_int == 0 and not n_then_n
    return ExampleShape(len(lines), len(first_tokens), first_int, all_numeric, n_then_n, multi_t)


def signature_compatible(sig: ProgramSignature, shape: ExampleShape) -> bool:
    """Cheap static compatibility between a program's I/O shape and the example's input layout."""
    if not sig.is_valid_syntax or sig.io_shape in ("unknown", "functional"):
        return False
    if sig.io_shape == "single-line":
        return shape.n_lines == 1
    if sig.io_shape == "multi-case-t":
        return shape.multi_case_t or (shape.first_token_int is not None and shape.n_lines > 1)
    if sig.io_shape == "n-then-n-lines":
        return shape.n_then_n_lines or shape.n_lines > 1
    return True  # token-line: no constraint


class UncertaintyRouter:
    """Decides which queries get corpus-wide verification from their dense score margin."""

    def __init__(self, tau: float = 0.05):
        self.tau = tau

    @staticmethod
    def margin(dense_scores: Dict[str, float]) -> float:
        if len(dense_scores) < 2:
            return float("inf")
        top = sorted(dense_scores.values(), reverse=True)[:2]
        return float(top[0] - top[1])

    def needs_corpus_wide(self, dense_scores: Dict[str, float]) -> bool:
        return self.margin(dense_scores) < self.tau


class SignatureGate:
    """Pre-computes corpus signatures once and filters candidates for a query's example shape."""

    def __init__(self, corpus: Dict[str, str]):
        self.signatures: Dict[str, ProgramSignature] = {doc_id: extract_program_signature(code) for doc_id, code in corpus.items()}
        self.eligible: List[str] = [d for d, s in self.signatures.items() if s.is_valid_syntax and s.io_shape not in ("unknown", "functional")]

    def candidates(self, stdin_text: str, exclude: Iterable[str] = ()) -> List[str]:
        shape = example_shape(stdin_text)
        skip = set(exclude)
        return [d for d in self.eligible if d not in skip and signature_compatible(self.signatures[d], shape)]

    def coverage(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for s in self.signatures.values():
            key = s.io_shape if s.is_valid_syntax else "syntax_error"
            out[key] = out.get(key, 0) + 1
        return out
