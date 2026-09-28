# Ablations — AppsRetrieval (CoIR), every row measured

Generated 2026-09-28 00:31 UTC by `vera/eval/ablation.py` from the MTEB `TaskResult` JSONs in `artifacts/` (test split, 3,765 queries, scored by `mteb.evaluate`) and the dev-split JSONs in `docs/` (500 held-out train pairs). Δ is versus the previous *measured* rung.

| Rung | Test NDCG@10 | Δ | Test MRR@10 | Test R@10 | Test R@100 | Dev NDCG@10 | Wall time | Notes |
|---|---|---|---|---|---|---|---|---|
| REF  lexical TF-IDF (word+bigram) | **2.62** |  | 2.16 | 4.17 | 17.58 | — | 19 s | reference only, never submitted |
| R0   gte-modernbert-base, 8192 ctx, stripped corpus | **56.83** |  | 52.07 | 71.90 | 91.87 | 71.43 | 15 s | dense chassis, zero-shot |
| R0b  R0 with raw (unstripped) corpus | *not run* | | | | | — | | preprocessing ablation |
| R0c  R0 with the example section dropped from the embedded query | *not run* | | | | | — | | query ablation |
| R1   R0 + LoRA fine-tune | *not run* | | | | | — | | requires GPU session |
| R2   R0/R1 + top-150 verification, rarity boost | **87.12** | +30.30 | 85.73 | 91.34 | 93.76 | 85.03 | 57 min | alpha fit on dev; alpha=3.0, top_k=150 |
| R3   R2 + corpus-wide verification behind gate | *not run* | | | | | — | | router tau fit on dev |
| R4   + QB-Norm demotion | *not run* | | | | | 85.11 | | beta fit on dev |

Reference points from the field (not ours): BM25 ≈ 0.95 NDCG@10 (CtrlFind), gte-modernbert-base zero-shot 56.4 (Granite-R2 paper, 1024-token cap) / 57.5 reproduced by a PRISM competitor.

## R2 alpha sweep (dev NDCG@10)

| alpha | 0.0 | 0.05 | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 | 0.35 | 0.4 | 0.5 | 0.75 | 1.0 | 1.5 | 2.0 | 3.0 | 5.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NDCG@10 | 71.43 | 72.66 | 74.69 | 76.31 | 77.98 | 79.13 | 80.58 | 81.74 | 82.58 | 83.52 | 84.52 | 84.95 | 84.99 | 84.97 | 85.03 | 84.91 |

Dev diagnostics: gold inside the dense top-150 for 485/500 queries; gold passes its own examples for 189; exactly one full passer in 148 queries; mean full passers per query 0.69.
