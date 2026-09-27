"""Worked-example parser, process-isolated execution sandbox, normalized comparator, rarity boost."""

from vera.verify.boost import (
    CandidateBoostInfo,
    GatedVerifier,
    QueryVerification,
    TopKVerifier,
    compute_rarity_confidence,
    min_max_normalize,
)
from vera.verify.comparator import ComparisonResult, compare_outputs, evaluate_comparison
from vera.verify.executor import (
    AdaptiveTimeoutTracker,
    CandidateVerificationResult,
    ExecutionResult,
    ExecutionSandbox,
    VerificationSandbox,
    detect_call_entry,
    run_isolated,
)
from vera.verify.parser import ExamplePair, ParseReport, WorkedExampleParser, statement_allows_any_order

__all__ = [
    "CandidateBoostInfo", "GatedVerifier", "QueryVerification", "TopKVerifier", "compute_rarity_confidence", "min_max_normalize",
    "ComparisonResult", "compare_outputs", "evaluate_comparison",
    "AdaptiveTimeoutTracker", "CandidateVerificationResult", "ExecutionResult", "ExecutionSandbox", "VerificationSandbox",
    "detect_call_entry", "run_isolated",
    "ExamplePair", "ParseReport", "WorkedExampleParser", "statement_allows_any_order",
]
