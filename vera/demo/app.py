"""
VERA demo surface (Gradio)
==========================
One page, three tabs:

1. **Retrieve** — query box (pre-loaded APPS statements) -> ranked snippets with badges
   ``PASSED 2/2 examples`` / ``FAILED`` / ``dense #k → #j`` / rarity confidence.
2. **Versions** — a stored path's lineage ``v1 → v2 → v3`` ranked working-first with
   ``behaviorally unchanged`` and ``fingerprint #…`` badges.
3. **Standing questions** — register queries, ingest a folder snapshot or git repo, watch
   which standing answers changed.

Run with ``python scripts/run_demo.py`` (``--encoder tfidf`` starts in seconds without the model).
"""

from __future__ import annotations

import html
from typing import Any, List, Optional

from vera.demo.backend import DemoBackend, Hit

BADGE_CSS = """
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:12px;margin-right:6px;font-family:ui-monospace,monospace}
.pass{background:#d1fae5;color:#065f46}.fail{background:#fee2e2;color:#7f1d1d}.info{background:#e0e7ff;color:#1e3a8a}
.fp{background:#dbeafe;color:#1e3a8a}.same{background:#ede9fe;color:#4c1d95}.rank{background:#fef3c7;color:#78350f}
pre{max-height:260px;overflow:auto;background:#0f172a;color:#e2e8f0;padding:10px;border-radius:8px;font-size:12px}
"""


def _badge(text: str) -> str:
    cls = "info"
    if text.startswith("PASSED"):
        cls = "pass"
    elif text.startswith("FAILED"):
        cls = "fail"
    elif text.startswith("fingerprint"):
        cls = "fp"
    elif text.startswith("behaviorally"):
        cls = "same"
    elif "→" in text:
        cls = "rank"
    return f'<span class="badge {cls}">{html.escape(text)}</span>'


def render_hits(hits: List[Hit], info: dict) -> str:
    parts = [f"<p><b>{info.get('examples_parsed', 0)}</b> worked example(s) parsed · {info.get('candidates_verified', 0)} candidates executed · "
             f"{info.get('full_passers', 0)} pass everything · {info.get('seconds', 0)} s</p>"]
    for h in hits:
        parts.append(f"<div><h4>#{h.final_rank} · <code>{html.escape(h.doc_id)}</code> · score {h.final_score:.3f}</h4>"
                     f"{''.join(_badge(b) for b in h.badges)}<pre>{html.escape(h.code[:2500])}</pre></div>")
    return "\n".join(parts)


def render_chain(ranked, chain) -> str:
    refs = {label: ref for label, ref, _ in chain}
    parts = ["<p>" + " → ".join(f"{label} <small>({refs[label][:10]})</small>" for label, _, _ in chain) + "</p>"]
    for r in ranked:
        code = next(c for label, _, c in chain if label == r.version_id)
        parts.append(f"<div><h4>rank {r.rank} · {r.version_id}</h4>{''.join(_badge(b) for b in r.badges)}"
                     f"<span class='badge info'>global {r.global_score:.3f} · diff {r.diff_score:.3f} · consensus {r.consensus:.2f}</span>"
                     f"<pre>{html.escape(code[:2500])}</pre></div>")
    return "\n".join(parts)


def build_app(backend: DemoBackend, example_queries: Optional[dict] = None):
    import gradio as gr

    examples = list((example_queries or {}).values())

    def do_retrieve(query: str, top_n: int):
        hits, info = backend.retrieve(query, top_n=int(top_n))
        return render_hits(hits, info)

    def do_ingest(source: str):
        rep = backend.ingest(source.strip())
        lines = [f"snapshots {rep['snapshots']} · new snippets {rep['new_snippets']} · unchanged {rep['unchanged_snippets']} · "
                 f"embedded {rep['embedded']} · store {rep['store']}"]
        for name, d in rep["standing_diffs"].items():
            flag = "CHANGED" if d["changed"] else "unchanged"
            lines.append(f"[{flag}] {name}: {d['top_paths']}" + (f"  (was {d['previous']})" if d["changed"] else ""))
        return "\n".join(lines), gr.update(choices=sorted(backend.store.chains))

    def do_chain(query: str, path: str):
        if not path:
            return "<p>pick a stored path</p>"
        ranked = backend.rank_chain(query, path)
        return render_chain(ranked, backend.version_chain(path))

    def do_register(name: str, query: str):
        backend.register_standing(name.strip() or f"q{len(backend.standing) + 1}", query)
        return "\n".join(f"{k}: {v[:80]}…" for k, v in backend.standing.items())

    with gr.Blocks(css=BADGE_CSS, title="VERA — verify-first code retrieval") as demo:
        gr.Markdown("# VERA — the encoder proposes, the runtime disposes\nDense retrieval over APPS solutions; the dense "
                    "top-K is *executed* on the statement's worked examples and re-ranked with a rarity-weighted boost.")
        with gr.Tab("Retrieve"):
            q = gr.Textbox(label="Problem statement", lines=10, value=examples[0] if examples else "")
            if examples:
                gr.Examples(examples=[[e] for e in examples], inputs=[q], label="Pre-loaded APPS statements")
            top_n = gr.Slider(3, 30, value=10, step=1, label="results")
            out = gr.HTML()
            gr.Button("Retrieve & verify", variant="primary").click(do_retrieve, [q, top_n], out)
        with gr.Tab("Versions"):
            vq = gr.Textbox(label="Problem statement the versions answer", lines=6, value=examples[0] if examples else "")
            path = gr.Dropdown(choices=sorted(backend.store.chains), label="stored path (ingest something first)")
            vout = gr.HTML()
            gr.Button("Rank versions").click(do_chain, [vq, path], vout)
        with gr.Tab("Standing questions"):
            with gr.Row():
                sname = gr.Textbox(label="name", scale=1)
                squery = gr.Textbox(label="standing query", lines=4, scale=3, value=examples[1] if len(examples) > 1 else "")
            reg = gr.Textbox(label="registered", lines=4)
            gr.Button("Register").click(do_register, [sname, squery], reg)
            src = gr.Textbox(label="folder snapshot or git repository path to ingest")
            ing = gr.Textbox(label="ingest report / standing-question diff", lines=8)
            gr.Button("Ingest & re-run standing questions", variant="primary").click(do_ingest, [src], [ing, path])
    return demo
