# VERA Milestone M4: Gold-Run Measurement & Gate Decision Log

**Evaluation date**: 2026-09-27  
**Sandbox**: process-isolated fork-per-run (`vera/verify/executor.py`), wall-clock timeout 0.75s, dual harness (script + call), normalized comparator  
**Rule**: no test-split program is executed here; the test split contributes statement *formats* only.

---

## 1. Formal gate decision

| Quantity | Value |
|---|---|
| Plan sample (300 random train pairs, seed 42): gold passes its own examples | **123/300 = 41.0%** |
| … of which statements with a parseable example | 141/300 = 47.0% |
| … pass rate given a parseable example | **123/141 = 87.2%** |
| Raw harness on the same sample (script mode, strict string equality, no py2/`return` rescue) | 70/300 = 23.3% |
| Harness gain (dual harness + normalized comparator + rescue) | **+17.7 pts** |
| **Projected gold-run rate on the test distribution** (format-weighted, §3) | **86.7%** |
| Wall time | plan sample 3.6s (12 ms/pair), full train pass 40s |

### Decision: `CORPUS_WIDE_UNLOCKED`
> Corpus-wide verification track (M8 / Task 09) is unlocked.  
> Gate policy (Plan.md M4): ≥60% → corpus-wide track lives · 40–60% → top-K only · <40% → cautious boost.

### Why the plan-sample number and the projection differ

The train partition is not distributed like the test split. Statement formats found by the parser:

| Format | Train pairs (all 5,000) | Test queries (3,765) |
|---|---|---|
| Codeforces `-----Examples-----` | 819 (16.4%) | 2947 (78.3%) |
| AtCoder/CodeChef `-----Sample Input-----` | 729 (14.6%) | 722 (19.2%) |
| LeetCode `Example N: Input/Output` | 731 (14.6%) | 39 (1.0%) |
| Bare `Sample Input/Output` markers | 24 (0.5%) | 7 (0.2%) |
| no parseable example | 2697 (53.9%) | 50 (1.3%) |

Over half of the train statements are LeetCode/Codewars-style function problems without a stdin example, while 98.7% of the test statements carry a Codeforces or AtCoder/CodeChef sample. The gate therefore has to be read on the format-weighted projection, not on the raw train sample.

---

## 2. Gold pass rate by statement format (all parseable train pairs)

| Format | Pairs | Gold passes all examples | Failure modes |
|---|---|---|---|
| Codeforces `-----Examples-----` | 819 | **711 (86.8%)** | wrong_output 91, error 11, timeout 3, empty_output 2, syntax_error 1 |
| LeetCode `Example N: Input/Output` | 731 | **502 (68.7%)** | wrong_output 116, error 67, syntax_error 38, empty_output 8 |
| AtCoder/CodeChef `-----Sample Input-----` | 729 | **684 (93.8%)** | wrong_output 34, error 6, empty_output 2, syntax_error 2, timeout 1 |
| Bare `Sample Input/Output` markers | 24 | **8 (33.3%)** | error 9, wrong_output 6, empty_output 1 |

Remaining failures on stdin-style formats are dominated by problems that accept **multiple valid answers** (the gold prints a different valid answer than the sample), statements whose sample lines were joined by the dataset export, and a handful of solutions that need more than the timeout on the sample. LeetCode failures are mostly `TreeNode`/`ListNode` inputs the call harness does not deserialize.

---

## 3. Projection to the test distribution

| Test format | Test queries | Share | Gold pass rate (from §2) | Contribution |
|---|---|---|---|---|
| Codeforces `-----Examples-----` | 2947 | 78.3% | 86.8% | 68.0 pts |
| AtCoder/CodeChef `-----Sample Input-----` | 722 | 19.2% | 93.8% | 18.0 pts |
| no parseable example | 50 | 1.3% | 0.0% | 0.0 pts |
| LeetCode `Example N: Input/Output` | 39 | 1.0% | 68.7% | 0.7 pts |
| Bare `Sample Input/Output` markers | 7 | 0.2% | 33.3% | 0.1 pts |
| **Total** | 3765 | 100% | | **86.7%** |

---

## 4. Consequences for downstream tasks

1. **Task 08 (R2, top-K boost)**: verification over the dense top-150 with the rarity-weighted boost; α fit on the 500-pair dev split.
2. **Task 09 (R3, corpus-wide)**: unlocked by this gate; must still win on dev against R2 to ship.
3. **History**: the first version of this document reported 6.0% (18/300). That number was an artifact of the previous parser matching the `-----Input-----`/`-----Output-----` *specification* sections as if they were the example, so gold programs received prose on stdin and crashed. It has been superseded by the measurements above.
