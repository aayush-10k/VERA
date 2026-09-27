"""
Stage-2 synthetic version benchmark (S1 rebuild time + S2 fingerprints + S3 ranking)
==================================================================================
Builds v1 -> v2 -> v3 histories over ~200 train problems whose statements carry a worked
example (test-like formats only):

    v1  gold with ONE injected token-level bug (must still compile)
    v2  the gold solution (the "fix")
    v3  v2 reformatted (comments / spacing) — behaviour identical

and measures, without touching the test split:

* S1 — incremental rebuild: snippets embedded by the version store vs a full re-embed at
  every snapshot, plus a 20-commit synthetic repo history (real wall time of the encoder).
* S2 — fingerprint certification: v3 certified *behaviorally unchanged* vs v2; v1 vs v2
  flagged as different, including on the **both-pass** subset where v1 still passes the
  worked example (the case a sample-only checker cannot see).
* S3 — working-version-first rate of :class:`VersionRanker` (a working version ranked #1),
  overall and on the both-pass subset, vs. a dense-only ranking.

Outputs ``docs/stage2-benchmark.md`` and ``docs/stage2_benchmark.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vera.data.loader import AppsRetrievalDataset
from vera.stage2.fingerprint import FingerprintIndex
from vera.stage2.ranker import VersionRanker
from vera.stage2.store import SnippetRecord, VersionStore
from vera.verify.executor import VerificationSandbox, _try_compile
from vera.verify.parser import WorkedExampleParser, statement_allows_any_order

DOCS = PROJECT_ROOT / "docs"

# (pattern, replacement) one-token mutations; applied to a single random occurrence
MUTATIONS: List[Tuple[str, str]] = [
    (r"<=", "<"), (r"(?<![<>=!])<(?![<=])", "<="), (r">=", ">"), (r"(?<![<>=!])>(?![>=])", ">="),
    (r"==", "!="), (r"\+ 1\b", "- 1"), (r"- 1\b", "+ 1"), (r"\+1\b", "-1"), (r"-1\b", "+1"),
    (r"range\(([A-Za-z_]\w*)\)", r"range(1, \1)"), (r"range\(([A-Za-z_]\w*)\)", r"range(\1 - 1)"),
    (r"\bmax\(", "min("), (r"\bmin\(", "max("), (r"//", "/"), (r"\band\b", "or"), (r"\bor\b", "and"),
]


def mutate_once(code: str, rng: random.Random) -> Optional[str]:
    """Apply one random applicable mutation to one random occurrence; return None if none compiles."""
    ops = MUTATIONS[:]
    rng.shuffle(ops)
    for pat, rep in ops:
        matches = list(re.finditer(pat, code))
        if not matches:
            continue
        m = rng.choice(matches)
        mutated = code[: m.start()] + re.sub(pat, rep, m.group(0), count=1) + code[m.end():]
        if mutated != code and _try_compile(mutated)[0] is not None:
            return mutated
    return None


def reformat(code: str, rng: random.Random) -> str:
    """Behaviour-preserving edit: header comment, blank lines, trailing comments on some lines."""
    lines = code.replace("\r\n", "\n").split("\n")
    out = ["# refactor: cosmetic clean-up (v3)", ""]
    for ln in lines:
        out.append(ln)
        if ln.strip() and not ln.rstrip().endswith((":", "\\", ",")) and rng.random() < 0.15 and "#" not in ln \
                and not ln.strip().startswith(("'", '"')):
            out[-1] = ln + "  # reviewed"
    return "\n".join(out) + "\n"


def make_encoder(kind: str, corpus_texts: Sequence[str]) -> Tuple[Callable[[Sequence[str]], np.ndarray], str]:
    if kind == "st":
        from vera.chassis.baseline import DenseChassis

        ch = DenseChassis(batch_size=8, show_progress=False)
        return (lambda texts: ch.encode(list(texts))), "gte-modernbert-base"
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(token_pattern=r"[A-Za-z_][A-Za-z0-9_]*|\d+|\S", ngram_range=(1, 2), sublinear_tf=True).fit(list(corpus_texts))
    return (lambda texts: vec.transform(list(texts)).toarray().astype(np.float32)), "tfidf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--encoder", choices=["tfidf", "st"], default="tfidf")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--repo-files", type=int, default=200)
    ap.add_argument("--repo-commits", type=int, default=20)
    ap.add_argument("--regressions", type=float, default=0.5, help="fraction of chains where the bug is the LAST version")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    ds = AppsRetrievalDataset()
    parser = WorkedExampleParser()
    pairs = [(q, t, c) for q, t, c in ds.get_train_pairs()
             if parser.parse_report(t).source_format in ("codeforces", "dashed_sample")]
    rng.shuffle(pairs)

    sandbox = VerificationSandbox(default_timeout=0.75, reduced_timeout=0.3, workers=args.workers)
    # ---- build histories -----------------------------------------------------------
    histories = []
    for qid, q_text, gold in pairs:
        if len(histories) >= args.problems:
            break
        if _try_compile(gold)[0] is None:
            continue
        examples = parser.parse_examples(q_text)
        gold_res = sandbox.verify_candidate(gold, examples, multiline_set=statement_allows_any_order(q_text))
        if not gold_res.all_passed:
            continue  # need a gold that is known-working on its sample
        bug = mutate_once(gold, rng)
        if bug is None:
            continue
        bug_res = sandbox.verify_candidate(bug, examples, multiline_set=statement_allows_any_order(q_text))
        regression = rng.random() < args.regressions
        # fix-chain: bug -> fix -> cosmetic refactor ; regression-chain: good -> cosmetic refactor -> bug introduced
        chain = [("v1", bug, True), ("v2", gold, False), ("v3", reformat(gold, rng), False)] if not regression else \
                [("v1", gold, False), ("v2", reformat(gold, rng), False), ("v3", bug, True)]
        histories.append({"qid": qid, "query": q_text, "chain": chain, "regression": regression,
                          "bug_passes_sample": bug_res.all_passed})
    n_reg = sum(h["regression"] for h in histories)
    print(f"[S3] built {len(histories)} histories ({n_reg} regression chains); bug passes sample in {sum(h['bug_passes_sample'] for h in histories)}", flush=True)

    encode_fn, enc_name = make_encoder(args.encoder, [c for h in histories for _, c, _ in h["chain"]] + [h["query"] for h in histories])

    # ---- S1: store + incremental rebuild ------------------------------------------------
    store = VersionStore()
    reports = []
    for step in range(3):
        recs = [SnippetRecord(path=f"{h['qid']}.py", code=h["chain"][step][1], source="folder", ref=f"v{step + 1}", timestamp=float(step)) for h in histories]
        reports.append(store.ingest(recs, snapshot=f"v{step + 1}"))
    t0 = time.perf_counter()
    n_inc = store.embed_missing(encode_fn)
    t_inc = time.perf_counter() - t0
    t0 = time.perf_counter()
    for step in range(3):
        encode_fn([h["chain"][step][1] for h in histories])
    t_full = time.perf_counter() - t0
    s1 = {"snapshots": [{"ref": r.snapshot, "records": r.records, "new": r.new_snippets, "unchanged": r.unchanged_snippets} for r in reports],
          "distinct_snippets_embedded": n_inc, "snippet_versions_total": 3 * len(histories),
          "incremental_embed_s": round(t_inc, 2), "full_reembed_s": round(t_full, 2),
          "speedup": round(t_full / max(t_inc, 1e-9), 2)}

    # synthetic repo history: N files, C commits each touching 1-3 files
    files = {f"f{i}.py": next(c for _, c, b in h["chain"] if not b) for i, h in enumerate(histories[: args.repo_files])}
    repo_store = VersionStore()
    repo_reports = [repo_store.ingest([SnippetRecord(p, c, "git", "c0", 0.0) for p, c in files.items()], "c0")]
    t_inc_repo = 0.0
    t0 = time.perf_counter(); repo_store.embed_missing(encode_fn); t_inc_repo += time.perf_counter() - t0
    t_full_repo = 0.0
    for c in range(1, args.repo_commits):
        for p in rng.sample(sorted(files), k=rng.randint(1, 3)):
            files[p] = mutate_once(files[p], rng) or reformat(files[p], rng)
        repo_reports.append(repo_store.ingest([SnippetRecord(p, code, "git", f"c{c}", float(c)) for p, code in files.items()], f"c{c}"))
        t0 = time.perf_counter(); repo_store.embed_missing(encode_fn); t_inc_repo += time.perf_counter() - t0
        t0 = time.perf_counter(); encode_fn(list(files.values())); t_full_repo += time.perf_counter() - t0
    s1["repo_history"] = {"files": len(files), "commits": args.repo_commits,
                          "snippets_embedded_incremental": sum(r.new_snippets for r in repo_reports),
                          "snippets_embedded_full": sum(r.records for r in repo_reports[1:]) + repo_reports[0].records,
                          "incremental_s": round(t_inc_repo, 2), "full_reembed_s": round(t_full_repo, 2),
                          "speedup": round((t_full_repo + 0.0) / max(t_inc_repo, 1e-9), 2)}

    # ---- S2 + S3: fingerprints and ranking ------------------------------------------------
    fpi = FingerprintIndex(sandbox=sandbox, n_probes=8)
    ranker = VersionRanker(encode_fn, sandbox=sandbox, fingerprints=fpi)
    q_embs = encode_fn([h["query"] for h in histories])
    stats = {"n": len(histories), "both_pass": 0, "regression_chains": sum(h["regression"] for h in histories),
             "refactor_certified_unchanged": 0, "bug_fingerprint_differs": 0, "bug_differs_on_both_pass": 0,
             "working_first": 0, "working_first_both_pass": 0, "no_consensus_working_first": 0, "no_consensus_working_first_both_pass": 0,
             "dense_only_working_first": 0, "dense_only_working_first_both_pass": 0,
             "recency_working_first": 0, "recency_working_first_both_pass": 0, "wall_s": 0.0}
    plain = VersionRanker(encode_fn, sandbox=sandbox, fingerprints=fpi, use_consensus=False)
    t0 = time.perf_counter()
    for i, h in enumerate(histories):
        chain = h["chain"]
        bug_id = next(vid for vid, _, b in chain if b)
        good_ids = {vid for vid, _, b in chain if not b}
        refactor_of = {"v3": "v2"} if not h["regression"] else {"v2": "v1"}
        ranked = ranker.rank(h["query"], [(vid, c) for vid, c, _ in chain], query_embedding=q_embs[i])
        by_id = {r.version_id: r for r in ranked}
        both = h["bug_passes_sample"]
        stats["both_pass"] += both
        ref, src = next(iter(refactor_of.items()))
        if by_id[ref].fingerprint and by_id[ref].fingerprint == by_id[src].fingerprint:
            stats["refactor_certified_unchanged"] += 1
        good_fp = by_id[src].fingerprint
        differs = by_id[bug_id].fingerprint != good_fp
        stats["bug_fingerprint_differs"] += differs
        stats["bug_differs_on_both_pass"] += differs and both
        wf = ranked[0].version_id in good_ids
        stats["working_first"] += wf
        stats["working_first_both_pass"] += wf and both
        plain_ranked = plain.rank(h["query"], [(vid, c) for vid, c, _ in chain], query_embedding=q_embs[i])
        pwf = plain_ranked[0].version_id in good_ids
        stats["no_consensus_working_first"] += pwf
        stats["no_consensus_working_first_both_pass"] += pwf and both
        dense_top = max(ranked, key=lambda r: (r.global_score, r.version_id)).version_id
        stats["dense_only_working_first"] += dense_top in good_ids
        stats["dense_only_working_first_both_pass"] += (dense_top in good_ids) and both
        stats["recency_working_first"] += "v3" in good_ids
        stats["recency_working_first_both_pass"] += ("v3" in good_ids) and both
        if (i + 1) % 50 == 0:
            print(f"  ranked {i + 1}/{len(histories)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    stats["wall_s"] = round(time.perf_counter() - t0, 1)
    sandbox.close()

    n, bp = stats["n"], max(1, stats["both_pass"])
    result = {"encoder": enc_name, "seed": args.seed, "s1": s1, "s2_s3": stats}
    DOCS.mkdir(exist_ok=True)
    (DOCS / "stage2_benchmark.json").write_text(json.dumps(result, indent=2))
    md = [
        "# Stage-2 synthetic version benchmark", "",
        f"Encoder for global/diff-line similarity: **{enc_name}** · {n} problems (train partition, Codeforces/AtCoder-style "
        f"statements with a worked example, gold verified working) · histories v1 (injected one-token bug) → v2 (fix) → v3 (cosmetic refactor). "
        f"Seed {args.seed}. No test-split data used.", "",
        "## S1 — incremental rebuild (content-addressed AST hashes)", "",
        "| Scenario | Snippet versions | Distinct snippets embedded | Incremental time | Full re-embed time | Speed-up |",
        "|---|---|---|---|---|---|",
        f"| v1→v2→v3 over {n} problems | {s1['snippet_versions_total']} | {s1['distinct_snippets_embedded']} | {s1['incremental_embed_s']} s | {s1['full_reembed_s']} s | **{s1['speedup']}×** |",
        f"| synthetic repo: {s1['repo_history']['files']} files × {s1['repo_history']['commits']} commits (1–3 files change per commit) | "
        f"{s1['repo_history']['snippets_embedded_full']} | {s1['repo_history']['snippets_embedded_incremental']} | "
        f"{s1['repo_history']['incremental_s']} s | {s1['repo_history']['full_reembed_s']} s | **{s1['repo_history']['speedup']}×** |", "",
        "v3 (reformatted) hashes to the same id as v2, so it is never re-embedded; in the repo scenario only the 1–3 touched files per commit are embedded.", "",
        "## S2 — behaviour fingerprints (8-probe battery mutated from the worked example)", "",
        "| Check | Rate |", "|---|---|",
        f"| cosmetic refactor certified *behaviorally unchanged* (identical fingerprint to its source version) | {stats['refactor_certified_unchanged']}/{n} = {100 * stats['refactor_certified_unchanged'] / n:.1f}% |",
        f"| buggy version's fingerprint differs from the working version's | {stats['bug_fingerprint_differs']}/{n} = {100 * stats['bug_fingerprint_differs'] / n:.1f}% |",
        f"| … on the **both-pass** subset (bug still passes the sample: {stats['both_pass']} problems) | {stats['bug_differs_on_both_pass']}/{stats['both_pass']} = {100 * stats['bug_differs_on_both_pass'] / bp:.1f}% |", "",
        "A sample-only checker sees no difference on the both-pass subset; the probe battery does in the fraction above.", "",
        f"## S3 — working-version-first ({stats['regression_chains']}/{n} chains are regressions: bug introduced in the LAST version)", "",
        "| Ranker | All problems | Both-pass subset |", "|---|---|---|",
        f"| **VERA VersionRanker** (execution separation → probe consensus → 0.7·global + 0.3·diff-line → trace tie-break) | {stats['working_first']}/{n} = **{100 * stats['working_first'] / n:.1f}%** | {stats['working_first_both_pass']}/{stats['both_pass']} = **{100 * stats['working_first_both_pass'] / bp:.1f}%** |",
        f"| same without probe consensus (execution separation → diff-line → trace) | {stats['no_consensus_working_first']}/{n} = {100 * stats['no_consensus_working_first'] / n:.1f}% | {stats['no_consensus_working_first_both_pass']}/{stats['both_pass']} = {100 * stats['no_consensus_working_first_both_pass'] / bp:.1f}% |",
        f"| dense similarity only | {stats['dense_only_working_first']}/{n} = {100 * stats['dense_only_working_first'] / n:.1f}% | {stats['dense_only_working_first_both_pass']}/{stats['both_pass']} = {100 * stats['dense_only_working_first_both_pass'] / bp:.1f}% |",
        f"| newest version first (recency prior) | {stats['recency_working_first']}/{n} = {100 * stats['recency_working_first'] / n:.1f}% | {stats['recency_working_first_both_pass']}/{stats['both_pass']} = {100 * stats['recency_working_first_both_pass'] / bp:.1f}% |", "",
        "Probe consensus is differential testing across the chain: two of three versions always share behaviour here (fix+refactor or good+refactor), "
        "so the odd one out is the bug whenever a probe exposes it; when no probe separates them (the remaining both-pass cases) the ranking falls back to text similarity and is a coin flip.", "",
        f"Wall time for fingerprinting + ranking: {stats['wall_s']} s.", "",
    ]
    (DOCS / "stage2-benchmark.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
