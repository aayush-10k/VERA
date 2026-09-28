"""
Launch the VERA demo page.

    python scripts/run_demo.py                      # gte-modernbert-base (uses cached corpus embeddings)
    python scripts/run_demo.py --encoder tfidf      # instant start, lexical chassis
    python scripts/run_demo.py --corpus-limit 2000  # smaller corpus for a laptop demo
    python scripts/run_demo.py --seed-versions      # create a v1/v2/v3 folder series and ingest it
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vera.demo.backend import DemoBackend, load_demo_corpus


def seed_version_folders(backend: DemoBackend, queries: dict) -> str:
    """Create a small v1 (buggy) / v2 (fixed) / v3 (refactored) folder series from real golds and ingest it."""
    import random

    from vera.data.loader import AppsRetrievalDataset
    from vera.stage2.ingest import ingest_folder_series

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from s3_version_benchmark import mutate_once, reformat  # type: ignore

    ds = AppsRetrievalDataset()
    qrels = ds.get_test_qrels()
    root = Path(tempfile.mkdtemp(prefix="vera_versions_"))
    rng = random.Random(3)
    for v in ("v1", "v2", "v3"):
        (root / v).mkdir()
    for qid in queries:
        gold = backend.raw_corpus[next(iter(qrels[qid]))]
        bug = mutate_once(gold, rng) or gold
        (root / "v1" / f"{qid}.py").write_text(bug)
        (root / "v2" / f"{qid}.py").write_text(gold)
        (root / "v3" / f"{qid}.py").write_text(reformat(gold, rng))
    ingest_folder_series(backend.store, [root / "v1", root / "v2", root / "v3"])
    backend.store.embed_missing(backend._encode)
    return str(root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["st", "tfidf"], default="st")
    ap.add_argument("--corpus-limit", type=int, default=None)
    ap.add_argument("--verify-top-k", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--seed-versions", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    corpus, queries = load_demo_corpus(limit=args.corpus_limit)
    backend = DemoBackend(corpus, encoder=args.encoder, verify_top_k=args.verify_top_k, alpha=args.alpha)
    if args.seed_versions:
        print("seeded version folders at", seed_version_folders(backend, queries))
    for i, (qid, text) in enumerate(list(queries.items())[:2]):
        backend.register_standing(f"standing-{qid}", text)

    from vera.demo.app import build_app

    demo = build_app(backend, example_queries=queries)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
