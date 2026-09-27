# Plan.md — VERA build plan

Main track M1→M12 is the critical path: finish M12 and the project is fully built and submitted.

S-track and E-track run independently — start them any time, in their own order, without blocking or being blocked by the main track (soft joins are marked).

## Track map

```
MAIN   M1 → M2 → M3 → M4 ⊗ → M5 → M6 → M7 → M8* → M9 → M10 → M11 → M12
                        │gate              (M8 only if M4 gate passed)
SIDE   S1 → S2 → S3 → S4          (S2 reuses M6's executor; stub it to start earlier)
EXT    E1 → E2 → E3               (any time; E1 as early as possible)

⊗ = go/no-go decision gate   * = conditional task
Joins: S3+S4 feed M11/M12 (PPT + demo). M9's ablation table feeds M11.
```

## Main track (sequence, each task ends in a named artifact)

### M1 — Environment + dataset audit (day 1)
Repo (private) + pinned env; load AppsRetrieval via mteb; verify 3,765 queries / 8,765 docs / 1 gold each; document the qN↔dN alignment finding and partition tags in `docs/dataset-audit.md` (never used in scoring); inventory meta_information/starter_code; re-measure the parseable-example rate (~78% claimed).

Exit: audit doc committed. → Artifact: `docs/dataset-audit.md`

### M2 — Harness proof-of-life
SearchProtocol wrapper + AbsEncoder fallback stub; run `mteb.evaluate` end-to-end with any model; produce a valid TaskResult JSON. This de-risks submission mechanics before anything clever exists — a perfect pipeline that can't emit the JSON is an eliminated pipeline.

Exit: JSON validates. → Artifact: throwaway JSON + `mtebio/` skeleton

### M3 — R0 baseline
gte-modernbert-base, full 8192 context, boilerplate-stripped corpus; measure on test.

Exit: NDCG@10 ≥ 0.575 (reproduce or beat the field's free baseline). → Artifact: JSON #1

### M4 — Gold-run measurement ⊗ (the go/no-go gate)
300 sampled train pairs: run gold on its own example, before and after harness fixes (dual harness, normalized comparator, all examples).

Gate: **≥60%** → corpus-wide track (M8) lives · **40–60%** → top-K verification only (skip M8) · **<40%** → verification demoted to cautious boost, R2 is the ceiling.

Exit: measured %, decision recorded. → Artifact: `docs/goldrun.md`

### M5 — Fine-tuned chassis (R1)
4,500/500 train/dev split; LoRA + contrastive fine-tune with near-dup hard negatives; bake-off vs Qwen3-Embedding-0.6B and jina-code-0.5b (int8); pick by dev NDCG@10; ONNX export; measure.

Exit: dev delta positive; test JSON. → Artifact: model + JSON #2

### M6 — Verification engine core
Parser (all examples) · dual harness · normalized comparator · forkserver pool · adaptive timeouts. Unit-tested against the M4 sample.

Exit: M4 sample gold-run rate reproduced through the engine. → Artifact: `verify/` green

### M7 — Top-K verification (R2 — the guaranteed fallback ship)
Boost over dense top-150 with the rarity-weighted calibrated boost, α fit on dev; measure.

Exit: dev delta positive vs M5. → Artifact: JSON #3

### M8 — Corpus-wide mode (R3) — only if M4 gate passed
Signature extractor + layout gate over all 8,765 · uncertainty router · runtime budget check · dev-measure net vs M7 per tier; keep only what wins.

Exit: dev delta vs M7 decided (either direction is a valid outcome — record it).
→ Artifact: JSON #4 or a documented rejection

### M9 — Final calibration (R4)
QB-Norm demotion + ensemble weights on dev; regenerate the full ablation table; final test run.

Exit: `docs/ablations.md` complete, every row measured. → Artifact: final JSON

### M10 — Release
Pin seeds; fresh-clone reproduction rehearsal on a clean machine (CPU); tag GitHub release with the JSON attached; README with exact run steps and stated runtime.

Exit: teammate reproduces the JSON from a clean clone. → Artifact: tagged release

### M11 — PPT
Slides: 1 hook · 2 gate framing · 3 architecture ("the encoder proposes, the runtime disposes") · 4 benchmark findings disclosed (artifact analysis, incl. what we refused to use) · 5 ablation table verbatim · 6 P1 incremental numbers (from S-track) · 7 Bonus: both-pass version discrimination · 8 oracle-generalization slide (examples here; unit tests / doctests / signatures / logs in real repos) · 9 limits + open-risk line · 10 team + reproduction.

Exit: dry-run to the team survives the "did X actually help?" question for every slide.

### M12 — Demo video + hands-on rehearsal
Record live query → ranked, badged responses; version demo (v1→v3, buggy version drops, behaviorally-unchanged tag); standing-questions live diff. Rehearse the hands-on: they run the code on fresh queries — practice exactly that.

Exit: video rendered; one full mock hands-on completed. **Project fully built.**

## S-track — Stage-2 / P1-Bonus module (independent; own sequence)

**S1 — Version store + ingestion:** content-hash store; ingest from git *and* folder snapshots; incremental re-embed (only changed hashes). Exit: rebuild-time numbers on the synthetic benchmark.

**S2 — Behavior-fingerprint index:** signature-conditioned probe batteries; fingerprints incl. crash signatures; "behaviorally unchanged" certification. (Reuses M6's executor — stub runner if starting before M6.) Exit: fingerprint agreement measured on synthetic versions.

**S3 — Version ranking:** execution separation of buggy/working + diff-line ranker for the both-pass case + trace-value tie-break. Exit: working-version-first rate ≥ the rival's 100%/22 parity, plus measured wins on both-pass pairs (the case they concede).

**S4 — Demo surface:** Gradio page + standing-questions watcher wired to S1–S3.
Exit: demo drives M11 slides 6–7 and M12 footage.

## E-track — external/admin (any time, zero build dependency)

**E1 — SPOC email (send day 1):** pipeline legality (SearchProtocol vs encoder-only) + the deadline and screening-cut count. The deadline answer sets the fork below.

**E2 — Competitive watch:** weekly re-scan of public PRISM repos (CtrlFind, the AST/"evolutionary" team, new entrants). Our repo stays private until M10.

**E3 — Logistics:** PPT template, team roles, release checklist, clean reproduction machine.

## Deadline fork (set by E1's answer; default to SHORT until known)

- **SHORT (<10 days left):** ship at M7 (skip M8) · S-track = S1 + minimal S3 (execution separation + diff-line only) · M11 slide 7 scoped down honestly.
- **NORMAL (≥3 weeks):** full plan as written, M8 included, all of S-track.

## Parallelization (team of 2–3)

Person A: M1→M2→M3→M5 (chassis line). Person B: M4→M6→M7→M8 (verification line; M4 can start the moment M1's loaders exist). Person C or slack time: S-track + E-track. Everyone converges at M9–M12.

## Definition of fully built

All five submission artifacts exist (final JSON · tagged release · reproducible README · PPT · demo video); the pipeline regenerates the final JSON from a clean CPU clone; the fallback encoder-only JSON is archived; P1 demo runs on both git and folder version sources; every ablation row carries a measured number; no scoring path reads IDs or partition metadata.
