# PPT outline (M11) — 10 slides, every claim backed by a file in this repo

Rule for the dry-run: each slide must survive "did X actually help?" with a measured number. Slides marked *pending* get their number from `docs/ablations.md` once the R0/R2 runs land; nothing is quoted from a competitor without saying so.

| # | Slide | Content (measured) | Source |
|---|---|---|---|
| 1 | Hook | Dense retrieval *guesses* from text; APPS statements carry an executable oracle in 98.67 % of test queries. "The encoder proposes, the runtime disposes." | `docs/dataset-audit.md` |
| 2 | Gate framing | Gold passes its own example in 87.2 % of parseable pairs (86.7 % projected on the test mix) → verification has headroom; the plan's go/no-go gate and what each branch would have meant. Show the 6 % → 87 % story as a parser bug found by measurement. | `docs/goldrun.md` |
| 3 | Architecture | L1 dense chassis (gte-modernbert-base, 8192 ctx, stripped corpus) → L2 process-isolated verification of the top-150 with the rarity-weighted bounded boost → MTEB `SearchProtocol`. Sandbox facts: fork-per-run, SIGKILL timeouts, rlimits, ~4–6 ms dispatch. | `README.md`, `vera/verify/executor.py` |
| 4 | Benchmark findings disclosed | (a) `partition` and `meta_information` columns exist on both corpus and queries — we never read them; (b) every corpus doc is the gold of exactly one query, train golds are distractors for test; (c) train ≠ test statement distribution (54 % of train has no stdin sample vs 1.3 % of test); (d) 98.3 % of test statements fit in 1,024 tokens, so long context is not the lever. | `docs/dataset-audit.md`, `docs/risk-register.md` |
| 5 | Ablation table, verbatim | REF TF-IDF 2.62 · R0 56.83 · R2 87.12 (+30.3) · R2+QB 87.72 (+0.6) · final gated+QB **88.74** (+1.0), each with MRR@10 / R@10 / R@100, dev NDCG@10 and wall time; α = 3.0, β = 0.75, τ = 0.02 all fit on dev; R1 shown as *not run*. | `docs/ablations.md` |
| 6 | P1 incremental index | Normalized-AST snippet ids: reformat/comment/docstring edits keep the id; git + folder ingestion; 13.2× faster incremental rebuild on the 200-file × 20-commit history. | `docs/stage2-benchmark.md` §S1 |
| 7 | Bonus: both-pass version discrimination | Probe-battery fingerprints certify refactors as *behaviorally unchanged* (100 %); on the both-pass subset the battery separates bug from fix in 35.6 % and the consensus ranker puts a working version first in 86.4 % (dense-only 78.0 %, newest-first 45.8 %); overall 95.5 % vs 80.0 %. Say out loud that the plan's diff-line blend was measured to hurt and dropped. | `docs/stage2-benchmark.md` §S2–S3 |
| 8 | Oracle generalisation | Examples here; in real repos the same runtime oracle is unit tests, doctests, type signatures, logs — the fingerprint index is the reusable piece. One slide, no numbers claimed. | — |
| 9 | Limits + open risks | Multiple-valid-answer problems cannot be certified by the sample; TreeNode/ListNode inputs not executed (1 % of test); R1 fine-tune not trained (GPU); encoder-only ruling pending (fallback path exists). | `docs/risk-register.md` |
| 10 | Team + reproduction | `pip install -r requirements.lock` → `python scripts/reproduce_submission.py`; cold ≈ 75 min of embedding + 60 min of sandboxed execution on 4 cores, warm re-blend in minutes; 53 tests. | `README.md` |

Demo footage (M12): Retrieve tab (badges), Versions tab (v1 buggy → v2 fixed → v3 refactored, `behaviorally unchanged`), Standing-questions tab (ingest a snapshot, watch the diff). `python scripts/run_demo.py --seed-versions`.
