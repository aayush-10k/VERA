# VERA Milestone M4: Gold-Run Measurement & Gate Decision Log

**Evaluation Date**: 2026-09-27  
**Dataset**: CoIR APPS Retrieval (Train Partition Sample)  
**Sample Size**: 300 deterministic pairs (seed 42)  
**Execution Environment**: Python 3.12 (CPU Sandbox)  

---

## 1. Executive Summary & Formal Gate Decision

| Metric | Measured Value | Benchmark Baseline |
|---|---|---|
| **Gold-Run Pass Rate (Dual Harness + Comparator)** | **6.00%** (18/300) | $\ge 40.0\%$ minimum target |
| **Raw Baseline Pass Rate (Naive String Match)** | **6.00%** (18/300) | Reference floor |
| **Harness Delta (Verification Gain)** | **+0.00%** | Normalization & Mode B |
| **Parseable Example Coverage** | **48.33%** (145/300) | $\ge 75.0\%$ target |
| **Pass Rate Given Parseable Examples** | **12.41%** (18/145) | Theoretical ceiling |
| **Benchmark Execution Time** | **9.92 s** (33.1 ms/pair) | $< 10$ ms/dispatch target |

### Formal Decision: `CAUTIOUS_BOOST_CEILING`
> **Policy Directive**: Demoted to cautious additive boost (pass rate 6.0% < 40%).  
> **Operational Impact**: R2 is the ceiling; apply conservative rarity weighting with small alpha.

---

## 2. Failure Mode Analysis

Out of 300 evaluated problem-solution pairs, failures were categorized as follows:

| Failure Category | Count | Percentage of Total | Root Cause & Mitigation |
|---|---|---|---|
| **Passed All Examples** | **18** | **6.0%** | Fully validated execution |
| **Output Mismatch** | 2 | 0.7% | Solution algorithmic divergence, differing output formatting, or multi-case variation |
| **Runtime Exception** | 111 | 37.0% | Missing standard input lines, EOFError on custom input loops, or recursion limit |
| **Missing Statement Examples** | 155 | 51.7% | Narrative-only statements or non-standard diagrammatic inputs |
| **Timeout (>=0.75s)** | 8 | 2.7% | Inefficient $O(N^2)$ algorithm or slow I/O |
| **Empty Output** | 0 | 0.0% | Program completed without writing to stdout |
| **Syntax Error** | 6 | 2.0% | Malformed Python 2 code or partial snippet in corpus |

---

## 3. Comparison of Raw Baseline vs Dual Harness

```
Raw Baseline:   [###                                               ] 6.0%
Dual Harness:   [###                                               ] 6.0%
```

The VERA Dual Harness (Mode A script execution + Mode B callable instantiation) combined with normalized output comparison (whitespace insensitivity, float tolerance $10^{-6}$, case-insensitive verdict matching) boosted valid executions by **+0.00%** over raw string comparison.

---

## 4. Operational Instructions for Downstream Tasks

1. **Task 08 (Top-K Verification & Rarity Boost R2)**:
   - Verification will be applied over the top-$K=150$ dense candidates.
   - Boost confidence formula:
     $$\text{conf}(d, q) = \left(\frac{e_{pass}}{E}\right) \cdot \frac{1}{1 + \log_2(m)}$$
   - Dense scores will be min-max normalized and blended with $\alpha$ calibrated on the 500-pair dev split.
2. **Task 09 (Corpus-Wide Gate R3)**:
   - Bypassed per M4 gate decision; carry forward calibrated R2 as guaranteed fallback ship.
