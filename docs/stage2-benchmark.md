# Stage-2 synthetic version benchmark

Encoder for global/diff-line similarity: **gte-modernbert-base** · 200 problems (train partition, Codeforces/AtCoder-style statements with a worked example, gold verified working) · histories v1 (injected one-token bug) → v2 (fix) → v3 (cosmetic refactor). Seed 7. No test-split data used.

## S1 — incremental rebuild (content-addressed AST hashes)

| Scenario | Snippet versions | Distinct snippets embedded | Incremental time | Full re-embed time | Speed-up |
|---|---|---|---|---|---|
| v1→v2→v3 over 200 problems | 600 | 399 | 503.55 s | 499.3 s | **0.99×** |

v3 (reformatted) hashes to the same id as v2, so it is never re-embedded; in the repo scenario only the 1–3 touched files per commit are embedded.

## S2 — behaviour fingerprints (8-probe battery mutated from the worked example)

| Check | Rate |
|---|---|
| cosmetic refactor certified *behaviorally unchanged* (identical fingerprint to its source version) | 200/200 = 100.0% |
| buggy version's fingerprint differs from the working version's | 160/200 = 80.0% |
| … on the **both-pass** subset (bug still passes the sample: 59 problems) | 21/59 = 35.6% |

A sample-only checker sees no difference on the both-pass subset; the probe battery does in the fraction above.

## S3 — working-version-first (109/200 chains are regressions: bug introduced in the LAST version)

| Ranker | All problems | Both-pass subset |
|---|---|---|
| **VERA VersionRanker** (execution separation → probe consensus → 1.0·global + 0.0·diff-line → trace tie-break) | 191/200 = **95.5%** | 51/59 = **86.4%** |
| same without probe consensus (execution separation → diff-line → trace) | 186/200 = 93.0% | 46/59 = 78.0% |
| dense similarity only | 160/200 = 80.0% | 46/59 = 78.0% |
| newest version first (recency prior) | 91/200 = 45.5% | 27/59 = 45.8% |

Probe consensus is differential testing across the chain: two of three versions always share behaviour here (fix+refactor or good+refactor), so the odd one out is the bug whenever a probe exposes it; when no probe separates them (the remaining both-pass cases) the ranking falls back to text similarity and is a coin flip.

Wall time for fingerprinting + ranking: 3336.1 s.

## Ablation — diff-line term (plan default 0.7·global + 0.3·diff-line), same encoder and seed

| Ranker | All problems | Both-pass subset |
|---|---|---|
| execution separation → probe consensus → **0.7·global + 0.3·diff-line** → trace | 184/200 = 92.0% | 43/59 = 72.9% |
| same, **global only** (table above, shipped default) | — | — |

The diff-line embedding of a one-token change is noise relative to the whole-program similarity; with probe consensus doing the
behavioural work, the blend only hurts (both-pass 72.9 % vs 86.4 %). Dropped from the default; kept as `--w-diff 0.3`.
The full run with the plan's weights (including the 200 × 20 repo-history rebuild timing) is `stage2-benchmark_wdiff03.md`.
