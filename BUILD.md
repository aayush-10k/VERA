# BUILD.md — VERA (Verify-first Retrieval Architecture)
Samsung PRISM GenAI Hackathon · Theme 1: Agentic Code Intelligence (CoIR APPS)

## 1. What this is, in plain terms
Every APPS problem statement carries its own worked example. We turn that example into an
executable test, let candidate programs actually take the test, and rank the ones that pass —
instead of only guessing by text similarity. A fine-tuned encoder supplies candidates and ranks
everything the test can't reach. A behavior-fingerprint index handles versions: two versions are
"the same" when they *do* the same thing, not when they *read* the same.

Three layers:
- **L1 Chassis** — fine-tuned small encoder (retrieval floor, biggest single scoring lever)
- **L2 Verification** — run candidates on the query's own example (precision lever)
- **L3 Behavior index** — fingerprints, version store, diff-line ranking (P1 + Bonus + demo)

## 2. Ship ladder (never be unsubmittable)
Every rung below emits a valid MTEB TaskResult JSON. Ship the highest rung that is measured
to win on the dev split by the deadline.

```
R0  gte-modernbert-base, full 8192 ctx, stripped corpus          (~0.57-0.60 expected)
R1  R0 + fine-tuned encoder (LoRA on 4.5K train pairs)           (target 0.65-0.75)
R2  R1 + top-K verification boost (K=150)                        (+3-8 expected)
R3  R2 + corpus-wide verification behind signature gate          (conditional on M4 gate)
R4  R3 + QB-Norm demotion + calibrated ensemble                  (final)
```

## 3. Hard rules (non-negotiable)
1. **Never** use document IDs, partition tags, or any dataset metadata in scoring. The qN↔dN
   alignment and the train/test partition are audited, documented in `docs/dataset-audit.md`,
   and never touched by the pipeline. Content-based signals only.
2. Repo stays **private** until the submission release tag.
3. All tuning on the 500-pair **dev split only**. Test split touched at milestone JSONs only.
4. A submittable JSON exists at every phase boundary (see ship ladder).
5. CPU-only inference paths everywhere; no network calls at inference; pinned dependencies.
6. Verification is always a **bounded boost**, never a hard reorder or hard filter.

## 4. Known numbers (shared intel — do not re-litigate, verify where marked)
| Fact | Value | Source |
|---|---|---|
| Test queries / corpus docs | 3,765 / 8,765 (5,000 train-partition) | dataset, competitor audit |
| Gold docs per query | exactly 1, binary qrels | dataset audit |
| BM25 baseline | ~0.0095 NDCG@10 | competitor-measured |
| gte-modernbert-base zero-shot | 56.4 published / 57.5 reproduced (1024-token cap) | Granite R2 paper / PRISM competitor |
| Generic small models | 5–15 NDCG@10 | Granite R2 table |
| CtrlFind (E5-base + top-50 execution) | 11.52 → 19.96 NDCG@10; 420 improved / 10 worsened | their README |
| CtrlFind recall@50 | 31% (their binding ceiling) | their README |
| Queries with parseable worked example | ~78% | CtrlFind, **re-verify in M1** |
| Gold fails its own example | ~27% raw (trailing ws, float fmt, multi-case suspected) | CtrlFind sample, **re-measure in M4** |
| Cost per candidate run (naive) | ~60 ms, interpreter startup dominated | CtrlFind |

## 5. Repo layout
```
vera/
  data/            # loaders: CoIR apps via mteb, codeparrot/apps for train statements
  chassis/         # preprocessing, embedding, fine-tune, ONNX export
  verify/          # example parser, harnesses, comparator, executor pool, boost
  gate/            # static input-signature extractor, uncertainty router
  stage2/          # version store, fingerprints, diff-line ranker, trace matcher
  mtebio/          # SearchProtocol wrapper, AbsEncoder fallback, JSON generation
  demo/            # gradio app, standing-questions watcher
  eval/            # dev-split harness, ablation table generator
  docs/            # dataset-audit.md, ablations.md, risk-register.md
  scripts/         # make-style entrypoints, one per Plan.md task
```

## 6. Environment
- Python 3.11 · `mteb` (v2.x) · `sentence-transformers` · `torch` (CPU wheels for inference)
- `onnxruntime` (int8 export for 0.5–0.6B candidates) · `peft` + GPU runtime on Colab for M5 only
- `datasets` (codeparrot/apps) · `numpy` (8.7K docs → brute-force matmul, no FAISS needed)
- Pin everything in `requirements.lock`; organizers must reproduce on CPU from a fresh clone.

## 7. Component specs

### 7.1 Data (`data/`)
- Load AppsRetrieval through `mteb.get_task("AppsRetrieval")` — never hand-rolled splits.
- Train-pair set: join the 5,000 train-partition solutions to their problem statements from
  `codeparrot/apps` train. Hold out 500 pairs → **dev split** (fixed seed, committed list).
- Audit script writes `docs/dataset-audit.md`: qrel structure, id-alignment check result,
  meta_information field inventory (source URL, starter_code), parseable-example rate.

### 7.2 Chassis (`chassis/`)
- **Preprocess (corpus)**: strip template/boilerplate blocks before embedding — fast-IO headers
  (`sys.stdin` aliasing, `threading`/`recursionlimit` preambles), duplicated contest scaffolds.
  Keep the stripped and raw texts both; verification runs on raw.
- **Preprocess (query)**: embed the full statement at 8192 tokens (the 1024 cap known in the
  field truncates the tail where examples sit). Variant to A/B on dev: statement with example
  blocks removed from the *embedded* text (examples still feed L2).
- **Base model**: `gte-modernbert-base` primary; bake-off vs `Qwen3-Embedding-0.6B` and
  `jina-code-embeddings-0.5b` (int8 ONNX) on dev before committing.
- **Fine-tune (M5, Colab GPU, offline only)**: LoRA r=16 α=32 on the encoder,
  MultipleNegativesRankingLoss, batch 32–64 (grad-accum as needed), lr 1e-4 (LoRA) / 2e-5 (head),
  3–5 epochs, eval dev NDCG@10 per epoch, keep best. **Hard negatives**: per query, top-20
  dense non-gold + members of the query's near-duplicate problem cluster. Starting
  hyperparameters, not gospel — dev decides.
- Export: merge LoRA → ONNX int8 for CPU inference; cache corpus embeddings to disk keyed by
  `(model_hash, preproc_hash)`.

### 7.3 Verification engine (`verify/`)
- **Parser**: extract *all* Input/Output example pairs from the statement (`Example`, `Sample
  Input/Output`, fenced blocks). Emit `[(stdin_text, expected_stdout)]`. Log parse-fail reasons.
- **Dual harness**: (a) script mode — subprocess, stdin piped; (b) call mode — AST-detect a
  `solve()`/`Solution` entrypoint and starter_code signatures, invoke with parsed args. A
  program passes if either harness matches.
- **Comparator (normalized)**: token-wise compare on whitespace-split output; float tolerance
  1e-6 (abs or rel) when both tokens parse as float; trailing-whitespace/newline insensitive;
  exact case by default, case-insensitive retry recorded at lower confidence.
- **Executor pool**: `multiprocessing` forkserver, `workers = cores`, interpreter pre-warmed —
  target ≤10 ms dispatch overhead per run. Isolation: no network, temp CWD, output size cap,
  memory cap via `resource`.
- **Timeout policy**: 0.75 s default; adaptive memory — a program with ≥2 timeouts anywhere
  drops to 0.3 s, ≥4 is skipped and recorded.
- **Rarity-weighted boost** (the safety valve that makes corpus-wide sane):
  after running input x on the candidate set, let m = number of programs whose output matches
  the expected output. `conf(d,q) = (examples_matched / examples_total) · 1/(1 + log2(m))`,
  capped at 1. Final score = `norm(dense) + α·conf`, α grid-fit on dev. A bare "4" that half
  the pool prints is worth almost nothing; a 47-line exact match is near-proof. Never override
  a decisive dense signal; never hard-filter non-passers.

### 7.4 Corpus-wide gate (`gate/`) — active only if M4 gate passes
- **Signature extractor** (static AST): count `input()` / `sys.stdin` / `readline` usage,
  detect `t = int(input())`-then-loop shape; classify {single-line, n-then-n-lines,
  token-line, multi-case-t, unknown}. Robust to `input` rebinding (scan `sys.stdin` too).
  `unknown` → excluded from corpus-wide, top-K path only.
- **Uncertainty router**: if dense margin (score₁ − score₂) is above a dev-fit threshold, skip
  verification for that query entirely (saves runtime where the answer is already decisive).
- Runtime budget check before full test run: projected = Σ(candidates × per-run ms + timeout
  tail); must fit overnight on the build machine, else tighten gate.

### 7.5 QB-Norm demotion (`chassis/qbnorm.py`)
Querybank normalization with the **5,000 public train statements** as the bank: a corpus doc
sitting unusually close to known train statements gets its score normalized down. Content-based,
citable, and it captures the same effect as partition-tag demotion without touching metadata.
Fit strength on dev; keep an ablation row.

### 7.6 MTEB integration (`mtebio/`)
- Primary: model class implementing MTEB v2 **SearchProtocol** — `index(corpus)` embeds and
  caches; `search(queries, top_k)` runs dense → gate → verify → boost → returns score dicts.
- Fallback: pure `AbsEncoder` subclass (chassis only), kept warm in case organizers rule
  encoder-only. Both paths emit TaskResult JSON via `mteb.evaluate`.
- Known gotchas (from the field): ModelMeta needs a loader and `memory_usage_mb`; JSON dump
  needs `default=str` for datetimes.

### 7.7 Stage-2 / behavior index (`stage2/`)
- **Version store**: snippet id = sha256 of normalized AST dump (comments/whitespace/line
  numbers stripped) → moved or reformatted code keeps its id; only genuine AST change triggers
  re-embed. Ingest from **git** (per-commit blobs) and **folder snapshots** — organizers never
  specified the version supply format.
- **Behavior fingerprint**: per signature class, a deterministic probe battery (seeded inputs
  satisfying the shape and, when parseable, the stated constraints); fingerprint = sha256 of
  concatenated normalized outputs; crash → `CRASH:<ExcType>` token (still a fingerprint).
  Identical fingerprint across versions ⇒ certified behaviorally unchanged ⇒ skip re-ranking.
- **Diff-line ranker** (the rival's conceded case — two versions both pass the example):
  unified diff → embed only added/changed lines → blend `0.7·global + 0.3·diff` (weights fit
  on synthetic version dev-set).
- **Trace-value matcher** (narrow tier): when the statement contains ≥3 distinctive literal
  numbers, `sys.settrace` capture of int/str locals during the sample run; Jaccard overlap vs
  statement numbers as a tie-break for multi-valid-output problems.
- **Synthetic version benchmark**: generate v1/v2/v3 histories (real edits + injected one-token
  bugs) over ~200 problems to measure P1 rebuild time and working-version-first rate.

### 7.8 Demo (`demo/`)
Single Gradio page: query box → ranked snippets with badges (`PASSED 2/2 examples`,
`fingerprint #a3f…`, version chain v1→v3, `behaviorally unchanged` tags). Standing-questions
watcher: registered queries re-run on ingest events, answer-diff displayed live. Hands-on
requires *showing responses for a given query*, not numbers — this page is that.

## 8. Measurement protocol
- Every layer lands with a dev-split delta or it doesn't ship (drop what doesn't win — and say
  so in the PPT; measured rejection reads as rigor).
- `eval/ablation.py` regenerates `docs/ablations.md`: one row per rung R0–R4 plus each L2/L3
  component, columns NDCG@10 / MRR@10 / Δ / runtime. This table is PPT slide 5, verbatim.
- Milestone test-JSONs: after M3, M5, M7, M8, M9 only.

## 9. Risk register (`docs/risk-register.md`)
| Risk | Trigger | Mitigation | Owner task |
|---|---|---|---|
| Gold-run rate low | M4 < 40% post-fixes | verification demoted to cautious top-K boost (R2 ceiling) | M4 |
| Impostor passers | low-entropy outputs | rarity weighting + multi-example conjunction + boost cap | M7/M8 |
| Runtime blowup | projected > overnight | tighten gate, raise router threshold, adaptive timeouts | M8 |
| Encoder-only ruling | SPOC reply | AbsEncoder fallback path, chassis-only JSON | E1 + 7.6 |
| FT pulls train solutions up | dev regression on near-dup queries | QB-Norm + near-dup hard negatives + dev watch | M5/M9 |
| Rival copies corpus-wide | public repo activity | repo private till tag; speed; FT + fingerprints are slow to copy | E2 |
| Organizers can't reproduce | fresh-clone failure | pinned deps, CPU paths, README rehearsal on a clean machine | M10 |

## 10. Submission checklist (from the problem statement)
- [ ] `appsretrieval_results.json` from `mteb.evaluate` on the test split (final rung)
- [ ] GitHub release with the JSON attached as a release artifact
- [ ] README with exact fresh-clone → JSON reproduction steps (CPU, expected runtime stated)
- [ ] PPT (structure in Plan.md M11)
- [ ] Demo video showing live responses for given queries + P1/Bonus flow
- [ ] Encoder-only fallback JSON generated and archived (not submitted unless ruled necessary)
