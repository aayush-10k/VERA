"""
VERA Milestone M4: Gold-Run Measurement & Go/No-Go Decision Gate
================================================================
Executes 300 sampled gold solutions on their own worked examples to measure:
1. Baseline raw execution pass rate (naive script exec, strict equality)
2. Dual harness + normalized comparator pass rate
3. Granular error breakdown across failure modes
4. Formally applies the M4 Gate Policy and writes docs/goldrun.md
"""

from __future__ import annotations

import io
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vera.data.loader import AppsRetrievalDataset
from vera.verify.comparator import compare_outputs
from vera.verify.executor import VerificationSandbox
from vera.verify.parser import WorkedExampleParser

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
GOLDRUN_MD = DOCS_DIR / "goldrun.md"


def run_raw_baseline(code: str, stdin_str: str, expected_stdout: str, sandbox: VerificationSandbox) -> bool:
    """Naive baseline execution: direct execution under timeout, strict string comparison."""
    res = sandbox.execute_snippet(code, stdin_str, expected_output=None, force_timeout=0.35)
    if res.status == "ok" and res.stdout:
        return res.stdout.strip() == expected_stdout.strip()
    return False


def main():
    print("[M4 Gold-Run] Loading APPS dataset...", flush=True)
    dataset = AppsRetrievalDataset()
    train_pairs = dataset.get_train_pairs()
    print(f"[M4 Gold-Run] Available training pairs: {len(train_pairs)}", flush=True)

    # Sample exactly 300 deterministic pairs
    rng = random.Random(42)
    sampled_indices = sorted(rng.sample(range(len(train_pairs)), min(300, len(train_pairs))))
    sampled_pairs = [train_pairs[i] for i in sampled_indices]
    print(f"[M4 Gold-Run] Sampled {len(sampled_pairs)} pairs for evaluation.", flush=True)

    parser = WorkedExampleParser()
    sandbox = VerificationSandbox(default_timeout=0.50, reduced_timeout=0.25)

    raw_passes = 0
    dual_passes = 0
    parseable_count = 0
    failure_breakdown: Counter = Counter()

    print("[M4 Gold-Run] Executing gold-run benchmark...", flush=True)
    t0 = time.perf_counter()

    for idx, (qid, q_text, gold_code) in enumerate(sampled_pairs):
        if (idx + 1) % 50 == 0 or (idx + 1) == len(sampled_pairs):
            print(f"  Processed {idx + 1}/{len(sampled_pairs)} pairs... (dual passes: {dual_passes})", flush=True)

        examples = parser.parse_examples(q_text)

        if not examples:
            failure_breakdown["no_examples_in_statement"] += 1
            continue

        parseable_count += 1

        # 1. Raw baseline execution on all extracted examples
        raw_all_passed = True
        for ex in examples:
            if not run_raw_baseline(gold_code, ex.stdin, ex.expected_stdout, sandbox):
                raw_all_passed = False
                break
        if raw_all_passed:
            raw_passes += 1

        # 2. Dual harness + normalized comparator
        cand_res = sandbox.verify_candidate(gold_code, examples)
        if cand_res.all_passed:
            dual_passes += 1
        else:
            # Categorize primary failure mode
            first_fail = next((r for r in cand_res.results if not r.matched), None)
            if first_fail is not None:
                if first_fail.status == "syntax_error":
                    failure_breakdown["syntax_error"] += 1
                elif first_fail.status in ("timeout", "timeout_skipped"):
                    failure_breakdown["timeout"] += 1
                elif first_fail.status == "error":
                    failure_breakdown["runtime_exception"] += 1
                elif first_fail.status == "empty_output":
                    failure_breakdown["empty_output"] += 1
                else:
                    failure_breakdown["output_mismatch"] += 1

    t1 = time.perf_counter()
    total_time_s = t1 - t0

    total_n = len(sampled_pairs)
    raw_pass_rate = (raw_passes / total_n) * 100.0
    dual_pass_rate = (dual_passes / total_n) * 100.0
    parseable_rate = (parseable_count / total_n) * 100.0
    conditional_pass_rate = (dual_passes / parseable_count * 100.0) if parseable_count > 0 else 0.0

    # Decision Gate Policy
    if dual_pass_rate >= 60.0:
        gate_decision = "CORPUS_WIDE_UNLOCKED"
        gate_summary = "Corpus-wide verification (Task 09 / M8) is UNLOCKED (pass rate >= 60%)."
        gate_action = "Proceed with full corpus signature routing and gate filtering in Task 09."
    elif dual_pass_rate >= 40.0:
        gate_decision = "TOP_K_ONLY"
        gate_summary = f"Restricted to Top-K verification (pass rate {dual_pass_rate:.1f}% is in 40-60% range)."
        gate_action = "Restrict verification strictly to Top-K candidates (K=150 in Task 08); skip corpus-wide sweep (Task 09)."
    else:
        gate_decision = "CAUTIOUS_BOOST_CEILING"
        gate_summary = f"Demoted to cautious additive boost (pass rate {dual_pass_rate:.1f}% < 40%)."
        gate_action = "R2 is the ceiling; apply conservative rarity weighting with small alpha."

    print("\n" + "=" * 60)
    print("M4 GOLD-RUN RESULTS SUMMARY")
    print("=" * 60)
    print(f"Sampled Pairs evaluated:         {total_n}")
    print(f"Problems with Parseable I/O:     {parseable_count} ({parseable_rate:.1f}%)")
    print(f"Raw Baseline Pass Rate:          {raw_passes}/{total_n} ({raw_pass_rate:.1f}%)")
    print(f"Dual Harness Pass Rate:          {dual_passes}/{total_n} ({dual_pass_rate:.1f}%)")
    print(f"Conditional Pass Rate (of I/O):  {dual_passes}/{parseable_count} ({conditional_pass_rate:.1f}%)")
    print(f"Total Benchmark Time:            {total_time_s:.2f} s ({(total_time_s/total_n)*1000:.1f} ms/pair)")
    print(f"Decision Gate:                   {gate_decision}")
    print(f"Decision Action:                 {gate_action}")
    print("Failure Breakdown:", dict(failure_breakdown))
    print("=" * 60)

    # Generate docs/goldrun.md
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    report_content = f"""# VERA Milestone M4: Gold-Run Measurement & Gate Decision Log

**Evaluation Date**: 2026-09-27  
**Dataset**: CoIR APPS Retrieval (Train Partition Sample)  
**Sample Size**: {total_n} deterministic pairs (seed 42)  
**Execution Environment**: Python 3.12 (CPU Sandbox)  

---

## 1. Executive Summary & Formal Gate Decision

| Metric | Measured Value | Benchmark Baseline |
|---|---|---|
| **Gold-Run Pass Rate (Dual Harness + Comparator)** | **{dual_pass_rate:.2f}%** ({dual_passes}/{total_n}) | $\\ge 40.0\\%$ minimum target |
| **Raw Baseline Pass Rate (Naive String Match)** | **{raw_pass_rate:.2f}%** ({raw_passes}/{total_n}) | Reference floor |
| **Harness Delta (Verification Gain)** | **+{dual_pass_rate - raw_pass_rate:.2f}%** | Normalization & Mode B |
| **Parseable Example Coverage** | **{parseable_rate:.2f}%** ({parseable_count}/{total_n}) | $\\ge 75.0\\%$ target |
| **Pass Rate Given Parseable Examples** | **{conditional_pass_rate:.2f}%** ({dual_passes}/{parseable_count}) | Theoretical ceiling |
| **Benchmark Execution Time** | **{total_time_s:.2f} s** ({(total_time_s/total_n)*1000:.1f} ms/pair) | $< 10$ ms/dispatch target |

### Formal Decision: `{gate_decision}`
> **Policy Directive**: {gate_summary}  
> **Operational Impact**: {gate_action}

---

## 2. Failure Mode Analysis

Out of {total_n} evaluated problem-solution pairs, failures were categorized as follows:

| Failure Category | Count | Percentage of Total | Root Cause & Mitigation |
|---|---|---|---|
| **Passed All Examples** | **{dual_passes}** | **{dual_pass_rate:.1f}%** | Fully validated execution |
| **Output Mismatch** | {failure_breakdown.get('output_mismatch', 0)} | {failure_breakdown.get('output_mismatch', 0)/total_n*100:.1f}% | Solution algorithmic divergence, differing output formatting, or multi-case variation |
| **Runtime Exception** | {failure_breakdown.get('runtime_exception', 0)} | {failure_breakdown.get('runtime_exception', 0)/total_n*100:.1f}% | Missing standard input lines, EOFError on custom input loops, or recursion limit |
| **Missing Statement Examples** | {failure_breakdown.get('no_examples_in_statement', 0)} | {failure_breakdown.get('no_examples_in_statement', 0)/total_n*100:.1f}% | Narrative-only statements or non-standard diagrammatic inputs |
| **Timeout (>=0.75s)** | {failure_breakdown.get('timeout', 0)} | {failure_breakdown.get('timeout', 0)/total_n*100:.1f}% | Inefficient $O(N^2)$ algorithm or slow I/O |
| **Empty Output** | {failure_breakdown.get('empty_output', 0)} | {failure_breakdown.get('empty_output', 0)/total_n*100:.1f}% | Program completed without writing to stdout |
| **Syntax Error** | {failure_breakdown.get('syntax_error', 0)} | {failure_breakdown.get('syntax_error', 0)/total_n*100:.1f}% | Malformed Python 2 code or partial snippet in corpus |

---

## 3. Comparison of Raw Baseline vs Dual Harness

```
Raw Baseline:   [{'#' * int(raw_pass_rate / 2)}{' ' * (50 - int(raw_pass_rate / 2))}] {raw_pass_rate:.1f}%
Dual Harness:   [{'#' * int(dual_pass_rate / 2)}{' ' * (50 - int(dual_pass_rate / 2))}] {dual_pass_rate:.1f}%
```

The VERA Dual Harness (Mode A script execution + Mode B callable instantiation) combined with normalized output comparison (whitespace insensitivity, float tolerance $10^{{-6}}$, case-insensitive verdict matching) boosted valid executions by **+{dual_pass_rate - raw_pass_rate:.2f}%** over raw string comparison.

---

## 4. Operational Instructions for Downstream Tasks

1. **Task 08 (Top-K Verification & Rarity Boost R2)**:
   - Verification will be applied over the top-$K=150$ dense candidates.
   - Boost confidence formula:
     $$\\text{{conf}}(d, q) = \\left(\\frac{{e_{{pass}}}}{{E}}\\right) \\cdot \\frac{{1}}{{1 + \\log_2(m)}}$$
   - Dense scores will be min-max normalized and blended with $\\alpha$ calibrated on the 500-pair dev split.
2. **Task 09 (Corpus-Wide Gate R3)**:
   - {'Corpus-wide routing is authorized.' if dual_pass_rate >= 60.0 else 'Bypassed per M4 gate decision; carry forward calibrated R2 as guaranteed fallback ship.'}
"""

    with open(GOLDRUN_MD, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[M4 Gold-Run] Committed formal gate decision to {GOLDRUN_MD}")


if __name__ == "__main__":
    main()
