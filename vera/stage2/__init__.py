"""Stage-2 / P1: content-addressed version store, multi-source ingestion, behaviour fingerprints, version ranking."""

from vera.stage2.fingerprint import Fingerprint, FingerprintIndex, build_probe_battery, fingerprint_program
from vera.stage2.ingest import ingest_folder_series, ingest_folder_snapshot, ingest_git_repo, rebuild_benchmark
from vera.stage2.ranker import RankedVersion, VersionRanker, diff_lines, statement_numbers, trace_overlap
from vera.stage2.store import SnippetRecord, VersionStore, normalize_code, snippet_id

__all__ = [
    "Fingerprint", "FingerprintIndex", "build_probe_battery", "fingerprint_program",
    "ingest_folder_series", "ingest_folder_snapshot", "ingest_git_repo", "rebuild_benchmark",
    "RankedVersion", "VersionRanker", "diff_lines", "statement_numbers", "trace_overlap",
    "SnippetRecord", "VersionStore", "normalize_code", "snippet_id",
]
