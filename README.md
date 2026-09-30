# VERA — Verify-first Retrieval Architecture

*Samsung PRISM GenAI Hackathon · Theme 1: Agentic Code Intelligence (CoIR `AppsRetrieval`)*

**The encoder proposes, the runtime disposes.** Every APPS problem statement carries a worked
example. VERA turns that example into an executable test, runs the dense retriever's candidate
programs on it in a process-isolated sandbox, and re-ranks with a rarity-weighted, bounded boost.
A behaviour-fingerprint index handles code *versions*: two revisions are "the same" when they
**do** the same thing on a probe battery, not when they read the same.

> **Status (2026-09-28).** Every rung of the ship ladder is measured on the full 3,765-query test
> split through `mteb.evaluate`: dense chassis **0.5683** → top-150 verification **0.8712** → + QB-Norm
> **0.8772** → + gated corpus-wide extension **0.8874 NDCG@10** (`appsretrieval_results.json`).
> Every number below comes from a script in `scripts/` on this repository — see `docs/ablations.md`
> for the table and `TASKS.md` for the honest checklist (including which earlier claims were
> withdrawn and why). Not done: the R1 LoRA fine-tune (needs a GPU), the release tag, PPT and video.

---

## What is measured so far

| Component | Measurement | Where |
|---|---|---|
| Worked-example parser | **98.67 %** of the 3,765 test statements yield executable `(stdin, expected)` pairs (Codeforces 78.3 %, AtCoder/CodeChef 19.2 %, LeetCode 1.0 %) | `docs/dataset-audit.md` |
| Gold-run gate (M4) | gold passes its own example in **87.2 %** of parseable train pairs; **86.7 %** projected on the test format mix → corpus-wide track unlocked | `docs/goldrun.md` |
| Sandbox | fork-per-run isolated worker, SIGKILL timeouts, rlimits; ≈ 4–6 ms dispatch; 53 unit tests green | `vera/verify/executor.py`, `tests/` |
| MTEB harness | `VERASearchProtocol` runs inside `mteb.evaluate`; TF-IDF reference row NDCG@10 = **0.0262** (BM25 in the field ≈ 0.0095) | `artifacts/ref_tfidf_results.json` |
| Stage-2 store (P1) | incremental rebuild **12.3×** faster than full re-embed on a 200-file × 20-commit history with the gte encoder (13.2× with TF-IDF) | `docs/stage2-benchmark_wdiff03.md` |
| Stage-2 fingerprints + ranking (Bonus) | cosmetic refactors certified *behaviorally unchanged* 100 %; working-version-first **95.5 %** (dense-only 80.0 %, newest-first 45.5 %); on the both-pass subset **86.4 %** vs 78.0 %. The plan's 0.3 diff-line weight was measured to hurt (72.9 % both-pass) and dropped. | `docs/stage2-benchmark.md` |
| R0 dense baseline | **NDCG@10 0.5683** · MRR@10 0.5207 · R@10 0.719 · R@100 0.919 on the full 3,765-query test split via `mteb.evaluate` (published zero-shot 56.4; a competitor's 1024-token reproduction 57.5) | `artifacts/m3_r0_results.json` |
| **R2 top-150 verification + rarity boost** | **NDCG@10 0.8712** · MRR@10 0.8573 · R@10 0.913 · R@100 0.938 on the full test split (α = 3.0 by argmax on the 500-pair dev split; dev sweep 0.714 → 0.850; on the 236 dev queries that carry an example 0.598 → 0.886). 57 min of sandboxed execution for 3,765 × 150 candidates on 4 cores. | `artifacts/m7_r2_results.json`, `docs/dev_r2.json` |
| R2 + QB-Norm demotion | **NDCG@10 0.8772** · MRR@10 0.8621 · R@10 0.923. Hubness of each corpus doc to the bank of 5,000 public train statements (top-3 mean), β = 0.75 by argmax on dev with the dev statements removed from the bank (dev 0.848 → 0.886; test +0.6). Content only — the `partition` column is never read. | `artifacts/m9_r4_results.json`, `docs/dev_r4.json` |
| **Final = gated extension + QB-Norm** (current submission) | **NDCG@10 0.8874** · MRR@10 0.8703 · R@10 0.940. Uncertainty router (dense top-1/top-2 margin < τ = 0.02, fit on dev) extends verification from rank 150 to rank 500 through the static signature gate for 2,038 of 3,765 test queries (479k extra executions, 60 min on 4 cores). Dev: 0.8863 → 0.8939. | `artifacts/m8_r4_final_results.json`, `docs/dev_r4_final.json`, `appsretrieval_results.json` |

---

## Architecture

```
query statement ──► WorkedExampleParser ──► [(stdin, expected)]           (vera/verify/parser.py)
      │                                              │
      ▼                                              ▼
L1  DenseChassis: gte-modernbert-base, 8192 ctx,   L2  VerificationSandbox: fork-per-run, rlimits,
    stripped corpus, brute-force cosine                 dual harness (script / call), normalized
    (vera/chassis/baseline.py)                          comparator, adaptive timeouts
      │ top-150                                         (vera/verify/executor.py, comparator.py)
      └────────────► TopKVerifier: conf = (e_pass/E) · 1/(1+log2 m);  S = norm(dense) + α·conf
                     (vera/verify/boost.py)   α fit on the 500-pair dev split only
                          │
        R3: GatedVerifier extends to dense rank 1000 for uncertain queries, filtered by the
            static signature gate (vera/gate/router.py)
        R4: QB-Norm hubness demotion with the public train statements as bank (vera/chassis/qbnorm.py)
                          │
                          ▼
              MTEB v2 SearchProtocol  (vera/mtebio/search_protocol.py)  →  mteb.evaluate → TaskResult JSON

L3  Stage-2 / P1 (vera/stage2/): normalized-AST version store · git + folder ingestion ·
    probe-battery behaviour fingerprints ("behaviorally unchanged") · working-first version ranker
```

Hard rules (`BUILD.md` §3): no scoring path reads ids, `partition` or `meta_information`;
all tuning on the dev split; verification is a bounded additive boost, never a filter; CPU-only
inference, pinned dependencies.

---

## Reproducible Setup & Quickstart

### Prerequisites
* **OS**: Linux (Ubuntu 20.04/22.04 LTS recommended) or Windows with WSL2 (the execution sandbox utilizes POSIX `os.fork` and `resource` limits for low-latency sandboxing).
* **Python**: `3.11.x`
* **Hardware**: Standard 4-core+ CPU with ≥ 16 GB RAM (Zero GPU required for inference).

---

### Step 1: Clone Repository & Create Virtual Environment

```bash
git clone https://github.com/aayush-10k/VERA.git
cd VERA

python3.11 -m venv .venv
source .venv/bin/activate  # On Windows WSL / Linux
```

### Step 2: Install Pinned Dependencies

```bash
# Install exact pinned requirements with CPU-optimized PyTorch wheels
pip install --upgrade pip
pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Run Unit Test Suite

Verify all engine components (Chassis, AST Parser, Sandbox, Stage-2 Store, QB-Norm):

```bash
pytest tests/ -q
# Expected output: 53 passed in ~15-20s
```

---

### Alternative: Run via Docker (Zero Environment Setup)

If you prefer containerized reproduction without local Python configuration:

```bash
# 1. Build the Docker image
docker build -t vera:latest .

# 2. Run unit test suite inside container
docker run --rm vera:latest

# 3. Reproduce submission inside container (mounts outputs to local artifacts/)
docker run --rm -v $(pwd)/artifacts:/app/artifacts vera:latest python scripts/reproduce_submission.py

# 4. Launch interactive Gradio demo via Docker
docker run --rm -p 7860:7860 vera:latest python scripts/run_demo.py --server-name 0.0.0.0
# Or using docker-compose:
docker-compose up
```

---

## One-Command Submission Reproduction

To reproduce the official hackathon winning submission (`appsretrieval_results.json`) from scratch:

```bash
# Preview the reproduction pipeline command
python scripts/reproduce_submission.py --dry-run

# Run full reproduction pipeline (R4: Gated Extension + QB-Norm)
python scripts/reproduce_submission.py --workers 4

# Or evaluate specific rungs:
python scripts/reproduce_submission.py --rung R0  # Dense baseline only (0.5683)
python scripts/reproduce_submission.py --rung R2  # Top-150 sandboxed verification (0.8712)
python scripts/reproduce_submission.py --rung R3  # Gated extension (0.8822)
python scripts/reproduce_submission.py --rung R4  # Final submission (0.8874)
```

### Execution Cost & Caching
* **Cold-start cost (4-core CPU)**:
  * Model download (`Alibaba-NLP/gte-modernbert-base`): ~300 MB (~1 min)
  * Corpus embedding (8,765 documents): ~45 min (cached to disk under `vera/chassis/cache/`)
  * Test query embedding (3,765 queries): ~25 min (cached to disk)
  * R2 sandboxed candidate verification: ~57 min
  * R4 gated extension verification: ~60 min
* **Warm re-runs**: ~2–3 minutes once embeddings and verification results are cached.

---

## Interactive Gradio Demo Surface

To explore the live query retrieval interface, version lineage, and standing-questions watcher:

```bash
# Launch with full dense model:
python scripts/run_demo.py

# Or launch instant start using TF-IDF reference encoder (starts in <5 seconds):
python scripts/run_demo.py --encoder tfidf
```

Open `http://127.0.0.1:7860` in your browser.

---

## Milestone Verification Scripts

Every claim and metric is backed by an executable script that writes its corresponding artifact:

| Script | Output Artifact | Description |
|---|---|---|
| `scripts/m1_audit.py` | `docs/dataset-audit.md` | Audit dataset distributions, verify zero ID leakage |
| `scripts/m2_harness_test.py` | `artifacts/m2_proof_of_life.json` | Validate MTEB v2 `SearchProtocol` harness |
| `scripts/m3_baseline.py` | `artifacts/m3_r0_results.json` | Evaluate R0 zero-shot ModernBERT baseline |
| `scripts/m4_goldrun.py` | `docs/goldrun.md` | Gold-run verification gate (87.2% pass rate) |
| `scripts/m7_topk_verify.py` | `artifacts/m7_r2_results.json` | R2 top-150 verification with rarity boost (0.8712) |
| `scripts/m8_corpus_gate.py` | `artifacts/m8_r4_final_results.json` | R4 final pipeline with gated extension + QB-Norm (0.8874) |
| `scripts/m9_qbnorm.py` | `artifacts/m9_r4_results.json` | R4a QB-Norm demotion ablation |
| `python -m vera.eval.ablation` | `docs/ablations.md` | Re-generate verbatim ablation table |
| `scripts/s3_version_benchmark.py` | `docs/stage2-benchmark.md` | Benchmark Stage-2 AST version store (12.3× speedup) |
| `scripts/reproduce_submission.py` | `appsretrieval_results.json` | One-command full reproduction runner |
| `scripts/run_demo.py` | Live Gradio Web UI | Interactive demonstration application |

---

## Repository layout

```
vera/
  data/       loader (parquet, same ids/texts as mteb's CoIR-Retrieval/apps), fixed 4,500/500 dev split
  chassis/    preprocessing, DenseChassis (+ explicit tfidf reference), QB-Norm, LoRA trainer, negative mining
  verify/     WorkedExampleParser, VerificationSandbox, comparator, TopKVerifier / GatedVerifier
  gate/       static I/O signatures, example-shape gate, uncertainty router
  mtebio/     VERASearchProtocol (mteb v2), encoder-only fallback, TaskResult serializer/validator
  stage2/     version store, ingestion, fingerprints, version ranker
  demo/       Gradio surface + UI-agnostic backend
  eval/       dev-split metrics, ablation table generator
scripts/      one entrypoint per milestone (table above)
docs/         audit, gold-run gate, dev fits, ablations, risk register, SPOC inquiry
artifacts/    MTEB TaskResult JSONs per rung
tests/        pytest suite
```

## Known limitations

- The R1 LoRA fine-tune needs a GPU session and has not been trained; R2 currently sits on the zero-shot chassis.
- LeetCode-style problems whose inputs are `TreeNode`/`ListNode` literals are not executed by the call harness (1 % of test).
- Problems with several valid answers cannot be certified by the sample; the boost stays bounded so they are not penalised.
---

## Team & Submission Details

* **Event**: Samsung PRISM GenAI Hackathon
* **Theme**: Theme 1: Agentic Code Intelligence (CoIR `AppsRetrieval`)
* **Team Name**: CipherPol
* **Institution**: Thapar University
* **Representative / Lead**: Aayushmaan Singh Meyan
* **Submission Date**: 30th September 2026

---

License: MIT.

