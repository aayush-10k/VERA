# Stage-2 synthetic version benchmark

Encoder for global/diff-line similarity: **tfidf** · 200 problems (train partition, Codeforces/AtCoder-style statements with a worked example, gold verified working) · histories v1 (injected one-token bug) → v2 (fix) → v3 (cosmetic refactor). Seed 7. No test-split data used.

## S1 — incremental rebuild (content-addressed AST hashes)

| Scenario | Snippet versions | Distinct snippets embedded | Incremental time | Full re-embed time | Speed-up |
|---|---|---|---|---|---|
| v1→v2→v3 over 200 problems | 600 | 399 | 0.29 s | 0.49 s | **1.68×** |
| synthetic repo: 200 files × 20 commits (1–3 files change per commit) | 4000 | 238 | 0.08 s | 1.07 s | **14.19×** |

v3 (reformatted) hashes to the same id as v2, so it is never re-embedded; in the repo scenario only the 1–3 touched files per commit are embedded.

## S2 — behaviour fingerprints (8-probe battery mutated from the worked example)

| Check | Rate |
|---|---|
| cosmetic refactor certified *behaviorally unchanged* (identical fingerprint to its source version) | 197/200 = 98.5% |
| buggy version's fingerprint differs from the working version's | 157/200 = 78.5% |
| … on the **both-pass** subset (bug still passes the sample: 59 problems) | 20/59 = 33.9% |

A sample-only checker sees no difference on the both-pass subset; the probe battery does in the fraction above.

## S3 — working-version-first (109/200 chains are regressions: bug introduced in the LAST version)

| Ranker | All problems | Both-pass subset |
|---|---|---|
| **VERA VersionRanker** (execution separation → probe consensus → 0.7·global + 0.3·diff-line → trace tie-break) | 179/200 = **89.5%** | 39/59 = **66.1%** |
| same without probe consensus (execution separation → diff-line → trace) | 172/200 = 86.0% | 32/59 = 54.2% |
| dense similarity only | 103/200 = 51.5% | 33/59 = 55.9% |
| newest version first (recency prior) | 91/200 = 45.5% | 27/59 = 45.8% |

Probe consensus is differential testing across the chain: two of three versions always share behaviour here (fix+refactor or good+refactor), so the odd one out is the bug whenever a probe exposes it; when no probe separates them (the remaining both-pass cases) the ranking falls back to text similarity and is a coin flip.

Wall time for fingerprinting + ranking: 65.3 s.
