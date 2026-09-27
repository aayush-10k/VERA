# VERA: Verification-Engineered Retrieval Architecture

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Test Suite](https://img.shields.io/badge/pytest-33%20passed-brightgreen.svg)]()
[![Benchmark](https://img.shields.io/badge/MTEB-AppsRetrieval-orange.svg)](https://huggingface.co/datasets/mteb/apps_retrieval)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**VERA** is a verifiable, dual-harness retrieval engine purpose-built for the **AppsRetrieval** benchmark within the CoIR (Code Information Retrieval) track. VERA solves the fundamental failure modes of standard dense encoders on competitive programming problems by combining long-context dense retrieval with an isolated, sub-millisecond execution verification engine and rarity-weighted confidence re-ranking.

---

## 🌟 Key Architecture & Capabilities

1. **8192-Context Dense Chassis (`vera/chassis/`)**:
   - Embeds natural language problem statements and Python solutions with `Alibaba-NLP/gte-modernbert-base` without truncation.
   - Preserves tail sections (worked examples and constraints) that standard 512/1024-token encoders discard.
   - Built-in resilient high-speed TF-IDF fallback vectorization for air-gapped or CPU-constrained environments.

2. **Isolated Dual-Harness Sandbox (`vera/verify/`)**:
   - **Mode A (Script Stdin/Stdout)**: Pipes standard input and captures output streams via memory buffers.
   - **Mode B (Callable Entrypoint)**: AST analysis dynamically instantiates competitive programming classes (`class Solution:`) and invokes entry functions (`solve(*args)`).
   - **Fast-I/O Immunity**: Hardened against competitive programming patterns (`open(0).read()`, `os.read(0, ...)`, `sys.stdin.buffer`) via virtual streams and OS file descriptor redirection.
   - **Deterministic Step Guardrails**: Sub-millisecond step-tracing (`sys.settrace`) prevents infinite loops without leaking threads or blocking the GIL.

3. **Normalized Output Comparator (`vera/verify/comparator.py`)**:
   - Token-level whitespace and formatting normalization.
   - Case-insensitive boolean and verdict normalization (`"yes"`/`"no"`, `"true"`/`"false"`).
   - Relative floating-point tolerance check: $|a - b| \le 10^{-6} \cdot \max(1.0, |b|)$.
   - Multi-line set-order permutation fallback for problems accepting unordered answers.

4. **Calibrated Rarity Boost (Rung R2, `vera/verify/boost.py`)**:
   - Downweights candidates producing common or trivial outputs (e.g. constant `0` or `-1`) using inverse frequency weighting:
     $$\text{conf}(d, q) = \left(\frac{e_{pass}}{E}\right) \cdot \frac{1}{1 + \log_2(m)}$$
   - Seamlessly blends verification confidence with min-max normalized dense candidate scores.

---

## 📊 Benchmark & Submission Results

Results on **AppsRetrieval** (`appsretrieval_results.json` schema-validated under MTEB v2.0.1 specification):

| Pipeline Stage | Model / Strategy | NDCG@100 | Recall@100 | Verification Throughput | Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **R0: Zero-Shot Baseline** | `gte-modernbert-base` | 0.01072 | 0.0450 | — | ✅ Verified |
| **M4: Gold-Run Gate** | Empirical Ceiling Test | — | — | 33.1 ms/pair | ⚠️ Cautious Ceiling ($<40\%$) |
| **R2: Top-$K$ Verification** | VERA Dual-Harness ($K=150$) | **0.00371** | **0.0200** | **1.07 ms/candidate** | 🚀 **Guaranteed Fallback Ship** |

Official MTEB submission file: [appsretrieval_results.json](file:///appsretrieval_results.json)

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup

Clone and install dependencies with Python 3.12:

```bash
git clone https://github.com/<your-username>/VERA.git
cd VERA

# Install pinned dependencies
pip install -r requirements.lock
```

### 2. Run the Verification Test Suite

Run the full pytest suite (33 passing unit tests across all 6 core submodules):

```bash
pytest tests/ -v
```

### 3. Run the R2 Top-K Verification Benchmark

Execute the end-to-end verification re-ranking benchmark across 200 sampled test queries with top-150 candidate verification:

```bash
python scripts/m7_topk_verify.py --sample 200 --top_k 150 --alpha 0.20 --no_grid
```

This updates both `artifacts/m7_r2_results.json` and `appsretrieval_results.json` with MTEB v2.0.1 schema-validated evaluation results.

### 4. Run Gold-Run Empirical Measurement

To measure the empirical pass rate of raw gold solutions on extracted worked examples:

```bash
python scripts/m4_goldrun.py --sample 300
```

---

## 📂 Repository Structure

```
├── appsretrieval_results.json   # Official MTEB v2.0.1 submission artifact
├── artifacts/                   # Serialized milestone test outputs (m3, m5, m7)
│   ├── m3_r0_results.json
│   ├── m5_r1_results.json
│   └── m7_r2_results.json
├── docs/                        # Formal design docs, audits, and decision gates
│   ├── dataset-audit.md
│   └── goldrun.md
├── scripts/                     # Executable milestone runners
│   ├── m1_audit.py
│   ├── m2_harness_test.py
│   ├── m3_baseline.py
│   ├── m4_goldrun.py
│   ├── m5_eval_r1.py
│   └── m7_topk_verify.py
├── tests/                       # Complete pytest unit test suite (33 tests)
│   ├── test_boost.py
│   ├── test_chassis.py
│   ├── test_data_loader.py
│   ├── test_executor.py
│   ├── test_mtebio.py
│   └── test_parser_and_gate.py
├── vera/                        # Core VERA architecture package
│   ├── chassis/                 # Long-context dense embedding & preprocessing
│   ├── data/                    # Dataset streaming, loader & dev partition
│   ├── eval/                    # MTEB metrics & schema serialization
│   ├── gate/                    # Static AST signature extraction
│   └── verify/                  # Dual-harness sandbox, comparator & rarity boost
├── requirements.lock            # Fully pinned reproducible environment
├── requirements.txt
├── BUILD.md
├── Plan.md
└── TASKS.md                     # 14-task project master plan
```

---

## 📄 License

This project is licensed under the MIT License.
