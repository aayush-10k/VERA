"""
VERA Data Loader & Split Manager
================================
Handles downloading, caching, and loading of the CoIR AppsRetrieval dataset.
Provides strict partition isolation and guards against metadata leakage.
Supports both Polars and Pandas for robust parquet loading.
"""

from __future__ import annotations

import json
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Try polars first (blazing fast, no pyarrow requirement), fallback to pandas
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    import pandas as pd

DATASET_URLS = {
    "queries": "https://huggingface.co/datasets/CoIR-Retrieval/apps-queries-corpus/resolve/main/data/queries-00000-of-00001-37cf369ef664f0a0.parquet",
    "corpus": "https://huggingface.co/datasets/CoIR-Retrieval/apps-queries-corpus/resolve/main/data/corpus-00000-of-00001-279bc28cd351fa67.parquet",
    "test_qrels": "https://huggingface.co/datasets/CoIR-Retrieval/apps-qrels/resolve/main/data/test-00000-of-00001-832c77cf909fcbbc.parquet",
    "train_qrels": "https://huggingface.co/datasets/CoIR-Retrieval/apps-qrels/resolve/main/data/train-00000-of-00001-0394bf6678199f07.parquet",
}

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache"
DEV_SPLIT_FILE = Path(__file__).resolve().parent / "dev_split_ids.json"


@dataclass(frozen=True)
class QueryItem:
    query_id: str
    text: str


@dataclass(frozen=True)
class CorpusItem:
    doc_id: str
    text: str


def ensure_data_cached(cache_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Downloads and caches parquet files if not already present."""
    target_dir = cache_dir or DEFAULT_CACHE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    local_paths = {}
    for key, url in DATASET_URLS.items():
        file_path = target_dir / f"{key}.parquet"
        if not file_path.exists() or file_path.stat().st_size == 0:
            print(f"[VERA Loader] Downloading {key} from HuggingFace...")
            req = urllib.request.Request(url, headers={"User-Agent": "VERA-Agent/1.0"})
            with urllib.request.urlopen(req) as resp, open(file_path, "wb") as f:
                f.write(resp.read())
            print(f"[VERA Loader] Cached {key} ({file_path.stat().st_size / 1024:.1f} KB).")
        local_paths[key] = file_path

    return local_paths


class AppsRetrievalDataset:
    """Encapsulates CoIR APPS Retrieval data with quarantine guards."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.files = ensure_data_cached(self.cache_dir)
        self._load_data()

    def _read_table(self, file_path: Path):
        if HAS_POLARS:
            return pl.read_parquet(file_path)
        else:
            return pd.read_parquet(file_path)

    def _load_data(self) -> None:
        t_queries = self._read_table(self.files["queries"])
        t_corpus = self._read_table(self.files["corpus"])
        t_test_qrels = self._read_table(self.files["test_qrels"])
        t_train_qrels = self._read_table(self.files["train_qrels"])

        if HAS_POLARS:
            self.queries_dict: Dict[str, str] = dict(zip(t_queries["_id"].to_list(), t_queries["text"].to_list()))
            self.corpus_dict: Dict[str, str] = dict(zip(t_corpus["_id"].to_list(), t_corpus["text"].to_list()))
            self.test_qrels: Dict[str, Dict[str, int]] = self._parse_qrels_polars(t_test_qrels)
            self.train_qrels: Dict[str, Dict[str, int]] = self._parse_qrels_polars(t_train_qrels)
        else:
            self.queries_dict = dict(zip(t_queries["_id"].astype(str), t_queries["text"]))
            self.corpus_dict = dict(zip(t_corpus["_id"].astype(str), t_corpus["text"]))
            self.test_qrels = self._parse_qrels_pandas(t_test_qrels)
            self.train_qrels = self._parse_qrels_pandas(t_train_qrels)

    @staticmethod
    def _parse_qrels_polars(df: pl.DataFrame) -> Dict[str, Dict[str, int]]:
        qrels: Dict[str, Dict[str, int]] = {}
        for row in df.iter_rows(named=True):
            qid = str(row["query_id"])
            cid = str(row["corpus_id"])
            score = int(row.get("score", 1))
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][cid] = score
        return qrels

    @staticmethod
    def _parse_qrels_pandas(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        qrels: Dict[str, Dict[str, int]] = {}
        cols = df.columns
        q_col = "query_id" if "query_id" in cols else cols[0]
        c_col = "corpus_id" if "corpus_id" in cols else cols[1]
        s_col = "score" if "score" in cols else (cols[2] if len(cols) > 2 else None)

        for _, row in df.iterrows():
            qid = str(row[q_col])
            cid = str(row[c_col])
            score = int(row[s_col]) if s_col else 1
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][cid] = score
        return qrels

    def get_test_queries(self) -> Dict[str, str]:
        """Returns query dictionary for queries present in test qrels."""
        test_qids = set(self.test_qrels.keys())
        return {qid: text for qid, text in self.queries_dict.items() if qid in test_qids}

    def get_corpus(self) -> Dict[str, str]:
        """Returns full corpus dictionary (doc_id -> raw_solution_text)."""
        return self.corpus_dict

    def get_test_qrels(self) -> Dict[str, Dict[str, int]]:
        """Returns test ground truth qrels."""
        return self.test_qrels

    def get_train_pairs(self) -> List[Tuple[str, str, str]]:
        """Returns list of (query_id, query_text, gold_corpus_text) for training partition."""
        pairs = []
        for qid, rels in self.train_qrels.items():
            if qid in self.queries_dict:
                q_text = self.queries_dict[qid]
                for cid in rels.keys():
                    if cid in self.corpus_dict:
                        c_text = self.corpus_dict[cid]
                        pairs.append((qid, q_text, c_text))
        return pairs

    def get_or_create_dev_split(self, dev_size: int = 500, seed: int = 42) -> Dict[str, Any]:
        """
        Creates or loads the deterministic 4,500 train / 500 dev split.
        Saves split indices to dev_split_ids.json.
        """
        if DEV_SPLIT_FILE.exists():
            with open(DEV_SPLIT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)

        train_qids = sorted(list(self.train_qrels.keys()))
        rng = random.Random(seed)
        shuffled = list(train_qids)
        rng.shuffle(shuffled)

        dev_qids = sorted(shuffled[:dev_size])
        train_qids_split = sorted(shuffled[dev_size:])

        split_data = {
            "metadata": {
                "total_train_partition": len(train_qids),
                "train_count": len(train_qids_split),
                "dev_count": len(dev_qids),
                "seed": seed,
            },
            "dev_query_ids": dev_qids,
            "train_query_ids": train_qids_split,
        }

        with open(DEV_SPLIT_FILE, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=2)

        print(f"[VERA Loader] Created fixed dev split: {len(dev_qids)} dev queries, {len(train_qids_split)} train queries.")
        return split_data
