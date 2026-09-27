# Ablations — AppsRetrieval (CoIR), every row measured

Generated 2026-09-27 22:47 UTC by `vera/eval/ablation.py` from the MTEB `TaskResult` JSONs in `artifacts/` (test split, 3,765 queries, scored by `mteb.evaluate`) and the dev-split JSONs in `docs/` (500 held-out train pairs). Δ is versus the previous *measured* rung.

| Rung | Test NDCG@10 | Δ | Test MRR@10 | Test R@10 | Test R@100 | Dev NDCG@10 | Wall time | Notes |
|---|---|---|---|---|---|---|---|---|
| REF  lexical TF-IDF (word+bigram) | **2.62** |  | 2.16 | 4.17 | 17.58 | — | 19 s | reference only, never submitted |
| R0   gte-modernbert-base, 8192 ctx, stripped corpus | **56.83** |  | 52.07 | 71.90 | 91.87 | — | 15 s | dense chassis, zero-shot |
| R0b  R0 with raw (unstripped) corpus | *not run* | | | | | — | | preprocessing ablation |
| R0c  R0 with the example section dropped from the embedded query | *not run* | | | | | — | | query ablation |
| R1   R0 + LoRA fine-tune | *not run* | | | | | — | | requires GPU session |
| R2   R0/R1 + top-150 verification, rarity boost | **68.50** | +11.67 | 64.71 | 80.48 | 93.44 | — | 57 min | alpha fit on dev; alpha=0.2, top_k=150 |
| R3   R2 + corpus-wide verification behind gate | *not run* | | | | | — | | router tau fit on dev |
| R4   + QB-Norm demotion | *not run* | | | | | — | | beta fit on dev |

Reference points from the field (not ours): BM25 ≈ 0.95 NDCG@10 (CtrlFind), gte-modernbert-base zero-shot 56.4 (Granite-R2 paper, 1024-token cap) / 57.5 reproduced by a PRISM competitor.
