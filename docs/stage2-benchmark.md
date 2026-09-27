# Stage-2 synthetic version benchmark

Encoder for global/diff-line similarity: **tfidf** · 200 problems (train partition, Codeforces/AtCoder-style statements with a worked example, gold verified working) · histories v1 (injected one-token bug) → v2 (fix) → v3 (cosmetic refactor). Seed 7. No test-split data used.

## S1 — incremental rebuild (content-addressed AST hashes)

| Scenario | Snippet versions | Distinct snippets embedded | Incremental time | Full re-embed time | Speed-up |
|---|---|---|---|---|---|
| v1→v2→v3 over 200 problems | 600 | 399 | 0.31 s | 0.45 s | **1.43×** |
| synthetic repo: 200 files × 20 commits (1–3 files change per commit) | 4000 | 240 | 0.08 s | 1.06 s | **13.17×** |

v3 (reformatted) hashes to the same id as v2, so it is never re-embedded; in the repo scenario only the 1–3 touched files per commit are embedded.

## S2 — behaviour fingerprints (8-probe battery mutated from the worked example)

| Check | Rate |
|---|---|
| v3 certified *behaviorally unchanged* vs v2 (identical fingerprint) | 200/200 = 100.0% |
| v1 (bug) vs v2 fingerprint differs | 161/200 = 80.5% |
| … on the **both-pass** subset (bug passes the sample: 56 problems) | 19/56 = 33.9% |

A sample-only checker sees no difference on the both-pass subset; the probe battery does in the fraction above.

## S3 — working-version-first

| Ranker | All problems | Both-pass subset |
|---|---|---|
| VERA VersionRanker (execution separation → 0.7·global + 0.3·diff-line → trace tie-break) | 154/200 = **77.0%** | 11/56 = 19.6% |
| dense similarity only | 119/200 = 59.5% | 34/56 = 60.7% |

Wall time for fingerprinting + ranking: 53.7 s.
