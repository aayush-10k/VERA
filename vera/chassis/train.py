"""
VERA contrastive fine-tuning of the dense chassis (Rung R1, Task 04 / M5)
========================================================================
Offline, GPU (Colab) training entrypoint. Nothing here runs at inference time.

    python -m vera.chassis.train --device cuda --epochs 3 --batch-size 32 --lora

* Data: the 4,500 train-split pairs (the 500 dev ids in ``vera/data/dev_split_ids.json`` are
  held out and never trained on); query = preprocessed statement, positive = stripped gold.
* Hard negatives (``--negatives``): per query, top-k dense non-gold candidates mined with
  the base model (``vera/chassis/mine_negatives.py``) + members of the query's near-duplicate
  statement cluster; stored as a JSON produced by ``scripts/mine_hard_negatives.py``.
* Loss: MultipleNegativesRankingLoss (in-batch negatives + the mined hard negative).
* LoRA (``--lora``, r=16, alpha=32, dropout 0.05 on attention/MLP projections via PEFT);
  otherwise full fine-tuning at a lower LR.
* Dev NDCG@10 is computed after every epoch with the same brute-force search used at
  inference; the best epoch is kept. ``--merge`` merges LoRA into the base weights so the
  result loads as a plain SentenceTransformer (what ``DenseChassis`` expects).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from vera.chassis.preprocess import ChassisPreprocessor
from vera.data.loader import AppsRetrievalDataset


def build_training_examples(
    ds: AppsRetrievalDataset,
    dev_ids: Set[str],
    hard_negatives: Optional[Dict[str, List[str]]] = None,
    seed: int = 42,
) -> List[Tuple[str, str, Optional[str]]]:
    """(query, positive, hard_negative | None) for every train-split pair; dev ids excluded."""
    pre = ChassisPreprocessor()
    stripped = pre.process_corpus(ds.corpus_dict)
    rng = random.Random(seed)
    out: List[Tuple[str, str, Optional[str]]] = []
    for qid, rels in ds.train_qrels.items():
        if qid in dev_ids or qid not in ds.queries_dict:
            continue
        q_text = pre.process_queries({qid: ds.queries_dict[qid]})[qid]
        for cid in rels:
            neg = None
            cands = [c for c in (hard_negatives or {}).get(qid, []) if c in stripped and c != cid]
            if cands:
                neg = stripped[rng.choice(cands[:5])]
            out.append((q_text, stripped[cid], neg))
    return out


def dev_ndcg10(model, ds: AppsRetrievalDataset, dev_ids: List[str], batch_size: int = 16) -> float:
    """Brute-force dev evaluation with the same preprocessing as inference."""
    import numpy as np

    from vera.eval.metrics import compute_retrieval_metrics

    pre = ChassisPreprocessor()
    corpus = pre.process_corpus(ds.corpus_dict)
    doc_ids = sorted(corpus)
    d = model.encode([corpus[i] for i in doc_ids], batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True)
    q = model.encode([pre.process_queries({i: ds.queries_dict[i]})[i] for i in dev_ids], batch_size=batch_size,
                     normalize_embeddings=True, convert_to_numpy=True)
    sims = q @ d.T
    results = {}
    for i, qid in enumerate(dev_ids):
        top = np.argpartition(sims[i], -100)[-100:]
        results[qid] = {doc_ids[j]: float(sims[i, j]) for j in top}
    return compute_retrieval_metrics({i: ds.train_qrels[i] for i in dev_ids}, results)["ndcg_at_10"]


def train(args: argparse.Namespace) -> None:
    import torch
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    ds = AppsRetrievalDataset()
    split = ds.get_or_create_dev_split()
    dev_ids = split["dev_query_ids"]
    negatives = json.loads(Path(args.negatives).read_text()) if args.negatives else None
    triples = build_training_examples(ds, set(dev_ids), negatives, seed=args.seed)
    print(f"[train] {len(triples)} training pairs ({sum(1 for t in triples if t[2])} with a mined hard negative); dev={len(dev_ids)}")

    model = SentenceTransformer(args.model, device=args.device)
    model.max_seq_length = args.max_seq_length
    if args.lora:
        from peft import LoraConfig, TaskType

        model[0].auto_model = _apply_lora(model[0].auto_model, LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            target_modules=["Wqkv", "Wo", "Wi"],  # ModernBERT attention + MLP projections
        ))

    examples = [InputExample(texts=[q, p] + ([n] if n else [])) for q, p, n in triples]
    # MNRL needs a uniform number of texts per batch: group triples and pairs separately
    trip = [e for e in examples if len(e.texts) == 3]
    pair = [e for e in examples if len(e.texts) == 2]
    loaders = [DataLoader(x, shuffle=True, batch_size=args.batch_size, drop_last=True) for x in (trip, pair) if len(x) >= args.batch_size]
    loss = losses.MultipleNegativesRankingLoss(model)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    best, log = -1.0, []
    base = dev_ndcg10(model, ds, dev_ids, batch_size=args.eval_batch_size)
    log.append({"epoch": 0, "dev_ndcg10": base})
    print(f"[train] epoch 0 (base model) dev NDCG@10 = {base:.4f}")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.fit(train_objectives=[(dl, loss) for dl in loaders], epochs=1, warmup_steps=args.warmup_steps,
                  optimizer_params={"lr": args.lr}, show_progress_bar=True, use_amp=args.device.startswith("cuda"))
        score = dev_ndcg10(model, ds, dev_ids, batch_size=args.eval_batch_size)
        log.append({"epoch": epoch, "dev_ndcg10": score, "minutes": round((time.time() - t0) / 60, 1)})
        print(f"[train] epoch {epoch} dev NDCG@10 = {score:.4f} ({log[-1]['minutes']} min)")
        if score > best:
            best = score
            _save(model, out_dir / "best", merge_lora=args.merge and args.lora)
        (out_dir / "training_log.json").write_text(json.dumps({"args": vars(args), "log": log, "best_dev_ndcg10": best}, indent=2))
    print(f"[train] best dev NDCG@10 = {best:.4f}  (base {base:.4f}) -> {out_dir / 'best'}")


def _apply_lora(hf_model, config):
    from peft import get_peft_model

    peft_model = get_peft_model(hf_model, config)
    peft_model.print_trainable_parameters()
    return peft_model


def _save(model, path: Path, merge_lora: bool) -> None:
    if merge_lora and hasattr(model[0].auto_model, "merge_and_unload"):
        model[0].auto_model = model[0].auto_model.merge_and_unload()
    model.save(str(path))


def main() -> None:
    ap = argparse.ArgumentParser(description="VERA chassis fine-tuning (GPU)")
    ap.add_argument("--model", default="Alibaba-NLP/gte-modernbert-base")
    ap.add_argument("--output", default=str(PROJECT_ROOT / "models" / "vera_chassis_r1"))
    ap.add_argument("--negatives", default=None, help="JSON {qid: [doc_id, ...]} from scripts/mine_hard_negatives.py")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--merge", action="store_true", help="merge LoRA into the base weights when saving")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
