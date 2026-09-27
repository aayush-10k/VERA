"""
VERA Normalized Output Comparator
=================================
Compares candidate execution output with expected output using:
1. Whitespace and trailing newline insensitivity
2. Token-wise whitespace split
3. Floating-point tolerance (|a - b| <= tol * max(1.0, |b|))
4. Case-insensitive canonical matching for binary verdicts (YES/NO, True/False)
5. Multi-line set/permutation fallback when order is flexible
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ComparisonResult:
    matched: bool
    reason: str
    tokens_actual: int
    tokens_expected: int


def _is_float(token: str) -> bool:
    try:
        val = float(token)
        return not (math.isnan(val) or math.isinf(val))
    except ValueError:
        return False


def _tokens_equal(a: str, e: str, float_tol: float = 1e-6) -> bool:
    if a == e:
        return True

    # Case-insensitive check for boolean / verdict words
    a_lower = a.lower()
    e_lower = e.lower()
    if a_lower == e_lower and a_lower in ("yes", "no", "true", "false", "possible", "impossible"):
        return True

    # Numeric float comparison
    try:
        fa = float(a)
        fe = float(e)
        if math.isnan(fa) or math.isnan(fe) or math.isinf(fa) or math.isinf(fe):
            return False
        diff = abs(fa - fe)
        limit = float_tol * max(1.0, abs(fe))
        return diff <= limit
    except ValueError:
        pass

    return False


def compare_outputs(
    actual: str,
    expected: str,
    float_tol: float = 1e-6,
    multiline_set: bool = False,
) -> bool:
    """Compare actual execution output against expected benchmark output.

    Parameters
    ----------
    actual : str
        Raw stdout captured from candidate execution.
    expected : str
        Target benchmark output string.
    float_tol : float
        Relative floating-point comparison tolerance (default 1e-6).
    multiline_set : bool
        If True or fallback, tests if lines/tokens match under permutation.

    Returns
    -------
    bool
        True if outputs are functionally equivalent under competitive
        programming norms.
    """
    res = evaluate_comparison(actual, expected, float_tol=float_tol, multiline_set=multiline_set)
    return res.matched


def evaluate_comparison(
    actual: str,
    expected: str,
    float_tol: float = 1e-6,
    multiline_set: bool = False,
) -> ComparisonResult:
    """Evaluate comparison and return detailed diagnostic result."""
    actual_str = actual.strip()
    expected_str = expected.strip()

    # Exact string match
    if actual_str == expected_str:
        tokens = len(actual_str.split())
        return ComparisonResult(matched=True, reason="exact_match", tokens_actual=tokens, tokens_expected=tokens)

    # Empty match
    if not actual_str and not expected_str:
        return ComparisonResult(matched=True, reason="empty_both", tokens_actual=0, tokens_expected=0)
    if not actual_str or not expected_str:
        return ComparisonResult(
            matched=False,
            reason="empty_mismatch",
            tokens_actual=len(actual_str.split()),
            tokens_expected=len(expected_str.split()),
        )

    # Token-wise comparison
    act_tokens = actual_str.split()
    exp_tokens = expected_str.split()

    if len(act_tokens) == len(exp_tokens):
        all_matched = True
        for a, e in zip(act_tokens, exp_tokens):
            if not _tokens_equal(a, e, float_tol=float_tol):
                all_matched = False
                break
        if all_matched:
            return ComparisonResult(
                matched=True,
                reason="token_match",
                tokens_actual=len(act_tokens),
                tokens_expected=len(exp_tokens),
            )

    # Multi-line line-by-line fallback
    act_lines = [line.strip() for line in actual_str.splitlines() if line.strip()]
    exp_lines = [line.strip() for line in expected_str.splitlines() if line.strip()]

    if len(act_lines) == len(exp_lines) and len(act_lines) > 0:
        lines_matched = True
        for al, el in zip(act_lines, exp_lines):
            al_toks = al.split()
            el_toks = el.split()
            if len(al_toks) != len(el_toks):
                lines_matched = False
                break
            for a, e in zip(al_toks, el_toks):
                if not _tokens_equal(a, e, float_tol=float_tol):
                    lines_matched = False
                    break
            if not lines_matched:
                break
        if lines_matched:
            return ComparisonResult(
                matched=True,
                reason="line_token_match",
                tokens_actual=len(act_tokens),
                tokens_expected=len(exp_tokens),
            )

    # Permutation / set fallback (if enabled or multi-line order insensitivity)
    if multiline_set and len(act_lines) == len(exp_lines) and len(act_lines) > 1:
        sorted_act = sorted(act_lines)
        sorted_exp = sorted(exp_lines)
        if sorted_act == sorted_exp:
            return ComparisonResult(
                matched=True,
                reason="line_set_match",
                tokens_actual=len(act_tokens),
                tokens_expected=len(exp_tokens),
            )

    # Tokens multiset fallback if counts match
    if multiline_set and len(act_tokens) == len(exp_tokens) and len(act_tokens) > 1:
        if sorted(act_tokens) == sorted(exp_tokens):
            return ComparisonResult(
                matched=True,
                reason="token_multiset_match",
                tokens_actual=len(act_tokens),
                tokens_expected=len(exp_tokens),
            )

    return ComparisonResult(
        matched=False,
        reason="token_mismatch",
        tokens_actual=len(act_tokens),
        tokens_expected=len(exp_tokens),
    )
