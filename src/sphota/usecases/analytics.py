# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Topic analytics by prefix: ``sphota analyze file.csv`` → a standalone HTML report.

Grouping (``by``): ``theme`` (first code block, named by the group's own most confident text;
best for one domain such as support tickets, D-020), ``region`` (32 broad cross-language regions)
or ``l1`` (first code block with the automatic Wikipedia-derived labels).

The report needs no server and no JavaScript libraries: a prefix tree (L1 → L2) with
counts, bars, multilingual labels, sample texts and a language breakdown (if a language
column exists). ``min_count`` suppresses small groups (k-anonymity style) before rendering.
"""

from __future__ import annotations

import csv
import html
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sphota.api import Sphota, encode_batch, label

TEXT_COLUMNS = (
    "text",
    "body",
    "content",
    "message",
    "sentence",
    "question",
    "title",
    "description",
)


@dataclass
class TopicNode:
    prefix: str
    count: int = 0
    samples: list[str] = field(default_factory=list)
    children: dict[str, TopicNode] = field(default_factory=dict)
    langs: Counter = field(default_factory=Counter)
    name: str | None = None  # theme mode: the group's most confident text
    _best: float = -1.0

    def label(self, lang: str = "en") -> str:
        return self.name or _lab(self.prefix, lang)


@dataclass
class TopicReport:
    n: int
    roots: list[TopicNode]
    suppressed: int
    title: str = "Sphota topic report"

    def top(self, k: int = 10) -> list[tuple[str, int, str]]:
        return [(r.prefix, r.count, r.label()) for r in self.roots[:k]]

    def to_dict(self) -> dict[str, Any]:
        def node(t: TopicNode) -> dict[str, Any]:
            return {
                "prefix": t.prefix,
                "label": t.label(),
                "count": t.count,
                "samples": t.samples,
                "children": [node(c) for c in sorted(t.children.values(), key=lambda c: -c.count)],
            }

        return {"n": self.n, "suppressed": self.suppressed, "topics": [node(r) for r in self.roots]}


def read_texts(
    path: str | Path, column: str | None = None, limit: int | None = None
) -> tuple[list[str], list[str | None]]:
    """Read texts (and an optional language column) from CSV/TSV/JSONL/TXT."""
    path = Path(path)
    texts: list[str] = []
    langs: list[str | None] = []
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        texts = [t.strip() for t in path.read_text(encoding="utf-8").splitlines() if t.strip()]
        langs = [None] * len(texts)
    elif suffix in {".jsonl", ".ndjson"}:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            col = column or next((c for c in TEXT_COLUMNS if c in row), None)
            if col is None:
                raise ValueError(
                    f"no text column found in {path}; pass --column (keys: {list(row)})"
                )
            texts.append(str(row[col]))
            langs.append(row.get("lang") or row.get("language"))
    else:
        delim = "\t" if suffix == ".tsv" else ","
        with open(path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f, delimiter=delim)
            fields = [c.lower() for c in (r.fieldnames or [])]
            col = column or next(
                (r.fieldnames[fields.index(c)] for c in TEXT_COLUMNS if c in fields), None
            )  # type: ignore[index]
            if col is None:
                raise ValueError(
                    f"no text column found in {path}; pass --column (columns: {r.fieldnames})"
                )
            lang_col = next(
                (c for c in (r.fieldnames or []) if c.lower() in {"lang", "language"}), None
            )
            for row in r:
                t = (row.get(col) or "").strip()
                if t:
                    texts.append(t)
                    langs.append(row.get(lang_col) if lang_col else None)
    if limit:
        texts, langs = texts[:limit], langs[:limit]
    if not texts:
        raise ValueError(f"no texts found in {path}")
    return texts, langs


def _lab(key: str, lang: str = "en") -> str:
    if key.startswith("R") and "-" not in key and len(key) == 3:
        from sphota.regions import region_label

        return region_label(key)
    return label(key, lang)


def build_report(
    texts: Sequence[str],
    codes: Sequence[Sphota] | None = None,
    langs: Sequence[str | None] | None = None,
    min_count: int = 1,
    samples: int = 3,
    title: str = "Sphota topic report",
    by: str = "region",
) -> TopicReport:
    """Group texts into a two-level tree. ``by='region'`` (default): 32 cross-language regions,
    then L1 cells; ``by='l1'``: L1 cells, then L2 cells; ``by='theme'``: like ``l1``, but each
    group is named by its own most confident text instead of the general-purpose labels."""
    if by not in ("region", "l1", "theme"):
        raise ValueError(f"by must be 'theme', 'region' or 'l1', not {by!r}")
    codes = list(codes) if codes is not None else encode_batch(list(texts))
    langs = list(langs) if langs is not None else [None] * len(texts)
    roots: dict[str, TopicNode] = {}
    for t, c, lg in zip(texts, codes, langs):
        if by == "region":
            p1, p2 = c.region, str(c.code.prefix(1))
        else:
            p1, p2 = str(c.code.prefix(1)), str(c.code.prefix(min(2, c.depth)))
        n1 = roots.setdefault(p1, TopicNode(p1))
        n1.count += 1
        conf = c.confidence[0] if c.confidence else 0.0
        if by == "theme" and conf > n1._best:
            n1._best, n1.name = conf, t[:90]
        if lg:
            n1.langs[lg] += 1
        if len(n1.samples) < samples:
            n1.samples.append(t)
        if c.depth >= 2:
            n2 = n1.children.setdefault(p2, TopicNode(p2))
            n2.count += 1
            conf2 = c.confidence[1] if len(c.confidence) > 1 else 0.0
            if by == "theme" and conf2 > n2._best:
                n2._best, n2.name = conf2, t[:90]
            if len(n2.samples) < samples:
                n2.samples.append(t)
    suppressed = 0
    kept = []
    for r in roots.values():
        if r.count < min_count:
            suppressed += r.count
            continue
        r.children = {k: v for k, v in r.children.items() if v.count >= min_count}
        if min_count > 1:
            r.samples = []  # never show raw texts when suppression is requested
            r.name = None
            for ch in r.children.values():
                ch.samples = []
                ch.name = None
        kept.append(r)
    kept.sort(key=lambda r: -r.count)
    return TopicReport(len(texts), kept, suppressed, title)


def render_html(rep: TopicReport, lang: str = "en") -> str:
    esc = html.escape
    mx = max((r.count for r in rep.roots), default=1)
    rows = []
    for i, r in enumerate(rep.roots):
        pct = 100 * r.count / rep.n
        kids = "".join(
            f"<li><code>{esc(c.prefix)}</code> {esc(c.label(lang).split(' › ')[-1])}"
            f" <span class=n>{c.count}</span>"
            + "".join(f"<div class=s>“{esc(s[:160])}”</div>" for s in c.samples[:2])
            + "</li>"
            for c in sorted(r.children.values(), key=lambda c: -c.count)[:12]
        )
        langs = ", ".join(f"{k} {v}" for k, v in r.langs.most_common(6))
        samples = "".join(f"<div class=s>“{esc(s[:200])}”</div>" for s in r.samples)
        rows.append(f"""
<details {"open" if i < 3 else ""}>
<summary><code>{esc(r.prefix)}</code><span class=lab>{esc(r.label(lang))}</span>
<span class=bar><i style="width:{100 * r.count / mx:.1f}%"></i></span><span class=n>{r.count} · {pct:.1f}%</span></summary>
<div class=body>{samples}{f"<div class=langs>languages: {esc(langs)}</div>" if langs else ""}
<ul>{kids}</ul></div></details>""")
    sup = (
        f"<p class=note>{rep.suppressed} items in groups smaller than the privacy threshold were suppressed.</p>"
        if rep.suppressed
        else ""
    )
    data = json.dumps([{"prefix": r.prefix, "count": r.count} for r in rep.roots])
    themes = any(r.name for r in rep.roots)
    unit = "themes" if themes else "topics"
    sub = (
        "Messages grouped by meaning, in any language. Each theme is named after its most typical "
        "message; its code (e.g. <code>WJX1</code>) is a stable ID you can track over time. "
        "Grouping is automatic and not perfect: check the examples."
        if themes
        else "Topics by Sphota code (space s1). The same code means the same area of meaning, "
        "in any language."
    )
    return f"""<!doctype html><html lang="{esc(lang)}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(rep.title)}</title>
<style>
:root{{--bg:#0b1120;--card:#121a2e;--text:#e6eeff;--muted:#93a4c3;--a:#4dabf7;--b:#ff6b6b}}
@media (prefers-color-scheme: light){{:root{{--bg:#f6f8fc;--card:#fff;--text:#132036;--muted:#5b6b86}}}}
body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:980px;margin:0 auto;padding:32px 16px}}
h1{{margin:0 0 4px;font-size:28px}} .sub{{color:var(--muted);margin-bottom:24px}}
.stats{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:24px}}
.stat{{background:var(--card);border-radius:12px;padding:12px 16px;min-width:140px}}
.stat b{{display:block;font-size:24px}}
details{{background:var(--card);border-radius:12px;margin-bottom:10px;padding:10px 14px}}
summary{{cursor:pointer;display:grid;grid-template-columns:70px 1fr 180px 110px;gap:12px;align-items:center;list-style:none}}
summary::-webkit-details-marker{{display:none}}
code{{font-family:ui-monospace,Menlo,monospace;color:var(--a);font-weight:600}}
.bar{{height:8px;background:rgba(128,128,128,.18);border-radius:4px;overflow:hidden}}
.bar i{{display:block;height:100%;background:linear-gradient(90deg,var(--a),var(--b))}}
.n{{color:var(--muted);text-align:right;font-variant-numeric:tabular-nums}}
.body{{padding:8px 0 4px 82px}} .s{{color:var(--muted);font-style:italic;margin:2px 0}}
ul{{padding-left:18px}} li{{margin:6px 0}} li .n{{margin-left:6px}}
.langs,.note{{color:var(--muted);font-size:13px}}
@media (max-width:640px){{summary{{grid-template-columns:60px 1fr}} .bar,.n{{display:none}} .body{{padding-left:0}}}}
</style></head><body><main>
<h1>{esc(rep.title)}</h1>
<div class=sub>{sub}</div>
<div class=stats><div class=stat><b>{rep.n:,}</b>messages</div><div class=stat><b>{len(rep.roots):,}</b>{unit}</div>
<div class=stat><b>{sum(len(r.children) for r in rep.roots):,}</b>sub-{unit}</div></div>
{sup}{"".join(rows)}
<p class=note>Generated by <a href="https://github.com/thiruvikramanpravartak/sphota">Sphota</a>. {"Themes are named by their most confident text from your file." if any(r.name for r in rep.roots) else "Labels are automatic keyphrase/LLM summaries of each region."}</p>
<script type="application/json" id="data">{data}</script>
</main></body></html>"""


def analyze_file(
    path: str | Path,
    column: str | None = None,
    out: str | Path = "sphota-report.html",
    min_count: int = 1,
    lang: str = "en",
    limit: int | None = None,
    by: str = "region",
) -> dict[str, Any]:
    texts, langs = read_texts(path, column, limit)
    rep = build_report(
        texts, langs=langs, min_count=min_count, title=f"Topics in {Path(path).name}", by=by
    )
    Path(out).write_text(render_html(rep, lang), encoding="utf-8")
    return {"out": str(out), "n": rep.n, "n_l1": len(rep.roots), "report": rep}


def prefix_counts(codes: Sequence[str], level: int = 1) -> Counter:
    """Count codes by prefix at ``level`` (the one-liner for dashboards)."""
    c: Counter = Counter()
    for code in codes:
        parts = str(code).split("-")
        c["-".join(parts[:level])] += 1
    return c


def group_by_prefix(
    items: Sequence[Any], codes: Sequence[str], level: int = 1
) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = defaultdict(list)
    for it, code in zip(items, codes):
        out["-".join(str(code).split("-")[:level])].append(it)
    return dict(out)
