# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Multilingual labels for L1 and L2 cells (c-TF-IDF keyphrases per language).

    python -m sphota.build.labels --scale small [--release src/sphota/data/v1]

Labels are informative data (spec Part 3 §5) and may be rebuilt without changing any code.

Method, per language:
* each cell's members in that language are grouped by *source* (Wikipedia article or FLORES
  sentence), so one long article cannot dominate a label;
* Wikipedia titles count as part of their sentences' text;
* sparse cells borrow evidence from their 3 nearest L1 neighbours with weight 0.25;
* tokens present in > 12% of cells (per language) are treated as stopwords (data-driven,
  works for every script);
* score = (source-weighted frequency / cell weight) * log(1 + mean cell mass / cell count of token)
  (BERTopic-style c-TF-IDF); a keyphrase must come from >= 2 sources when the cell has >= 3.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from sphota.build.common import DATA_DIR, ROOT, log

CJK = {"zh", "ja"}
_EN_STOP = set(
    """the of and or in on at to for from by with as is are was were be been this that these those it its
he she they his her their them we you not but if then than also which who whom whose what when where while into
over under about after before between during through can could would should will may might must does did has
have had there here one two three first second new other some any all most more many much very same each both
only own however although such being since because until upon within without among across""".split()
)


def tokens(text: str, lang: str) -> list[str]:
    t = unicodedata.normalize("NFC", text.lower())
    if lang in CJK:
        chars = [c for c in t if unicodedata.category(c).startswith("L")]
        return ["".join(chars[i : i + 2]) for i in range(len(chars) - 1)]
    words = []
    for raw in t.split():
        w = "".join(ch for ch in raw if unicodedata.category(ch)[0] not in "PSZC")
        w = w.strip("'’-")
        if len(w) < 3 or not any(unicodedata.category(ch).startswith("L") for ch in w):
            continue
        if any(ch.isdigit() for ch in w) or w in _EN_STOP:
            continue
        words.append(w)
    return words + [f"{a} {b}" for a, b in zip(words, words[1:])]


def ctfidf(
    cells: dict[str, list[tuple[str, str, str]]],
    lang: str,
    neighbors: dict[str, list[str]] | None,
    top: int,
    min_sources: int,
    stop_frac: float = 0.12,
) -> dict[str, str]:
    """``cells``: key -> [(lang, text, source)]. Returns key -> label for one language."""
    own: dict[str, dict[str, set]] = {}
    nsrc: dict[str, int] = {}
    for key, items in cells.items():
        bysrc: dict[str, set] = defaultdict(set)
        for lg, text, src, *_ in items:
            if lg == lang:
                bysrc[src].update(tokens(text, lang))
        if len(bysrc) < min_sources:
            continue
        tok_src: dict[str, set] = defaultdict(set)
        for src, toks in bysrc.items():
            for w in toks:
                tok_src[w].add(src)
        own[key] = tok_src
        nsrc[key] = len(bysrc)
    if not own:
        return {}
    cell_df = Counter()
    for tok_src in own.values():
        cell_df.update(tok_src.keys())
    n_cells = len(own)
    stop = {w for w, c in cell_df.items() if c > stop_frac * n_cells and n_cells >= 20}
    mass = {k: sum(len(s) for s in v.values()) for k, v in own.items()}
    avg = float(np.mean(list(mass.values())))
    out = {}
    for key, tok_src in own.items():
        tf: Counter = Counter({w: float(len(s)) for w, s in tok_src.items()})
        weight = float(nsrc[key])
        for nb in (neighbors or {}).get(key, [])[:3]:
            if nb in own:
                for w, s in own[nb].items():
                    tf[w] += 0.25 * len(s)
                weight += 0.25 * nsrc[nb]
        need = 2 if nsrc[key] >= 3 else 1
        scored = []
        for w, f in tf.items():
            if w in stop or len(tok_src.get(w, ())) < need:
                continue
            if " " in w and any(p in stop for p in w.split()):
                continue
            score = (f / weight) * np.log(1 + avg / cell_df[w])
            if " " in w:
                score *= 1.2
            scored.append((score, w))
        scored.sort(reverse=True)
        chosen: list[str] = []
        for _, w in scored:
            if any(w in x or x in w for x in chosen):
                continue
            chosen.append(w)
            if len(chosen) == top:
                break
        if chosen:
            out[key] = ", ".join(chosen)
    return out


def build_labels(scale: str, release_dir: Path) -> dict:

    from sphota.build.release import embeddings, load
    from sphota.release import load_release
    from sphota.space import project, quantize

    load_release.cache_clear()
    rel = load_release(str(release_dir), verify=False)
    langs, dev, _test, wiki, _tat = load(scale)
    E = embeddings(scale)
    members1: dict[str, list] = defaultdict(list)
    members2: dict[str, list] = defaultdict(list)

    def add(codes: np.ndarray, rows: list[tuple[str, str, str, str]]):
        for row, c in zip(rows, codes):
            members1[f"{c[0]}"].append(row)
            members2[f"{c[0]}.{c[1]}"].append(row)

    wcodes = quantize(rel, project(rel, E["wiki"]), levels=2).indices
    rows = [
        (lg, f"{title}. {text}", f"w:{lg}:{title}", text)
        for lg, title, text in wiki[["lang", "title", "text"]].itertuples(index=False)
    ]
    add(wcodes, rows)
    for lg in langs:
        dcodes = quantize(rel, project(rel, E[("dev", lg)]), levels=2).indices
        add(dcodes, [(lg, t, f"f:{i}", t) for i, t in enumerate(dev[lg].tolist())])
    nb = (
        {str(i): [str(int(j)) for j in rel.l1_neighbors[i][:3]] for i in range(rel.sizes[0])}
        if rel.l1_neighbors is not None
        else None
    )
    labels: dict[str, dict[str, str]] = defaultdict(dict)
    for lg in langs:
        l1 = ctfidf(members1, lg, nb, top=3, min_sources=2)
        l2 = ctfidf(
            {k: v for k, v in members2.items() if len(v) >= 4}, lg, None, top=2, min_sources=3
        )
        for k, v in {**l1, **l2}.items():
            labels[k][lg] = v
        log(f"labels {lg}: L1 {len(l1)}  L2 {len(l2)}")
    # L1 cells with no label at all borrow their nearest labelled neighbour's (marked "~")
    for i in range(rel.sizes[0]):
        k = str(i)
        if k not in labels and nb:
            for j in rel.l1_neighbors[i]:
                if str(int(j)) in labels and "en" in labels[str(int(j))]:
                    labels[k] = {"en": "~ " + labels[str(int(j))]["en"]}
                    break
    examples = {}
    for k, items in members1.items():
        flores_en = [sent for lg, _, src, sent in items if lg == "en" and src.startswith("f:")]
        wiki_en = [sent for lg, _, src, sent in items if lg == "en" and src.startswith("w:")]
        other = [sent for _, _, _, sent in items]
        pool = sorted(flores_en, key=len) + sorted(wiki_en, key=len) or sorted(other, key=len)
        examples[k] = [x for x in pool if len(x) <= 160][:3] or pool[:3]
    # prompts material for LLM naming (build cache, not shipped)
    cache = DATA_DIR / scale / "build"
    cache.mkdir(parents=True, exist_ok=True)

    def sample(items, n):
        en = [it[3] for it in items if it[0] == "en"]
        rest = [it[3] for it in items if it[0] != "en"]
        pool = [x for x in en + rest if len(x) <= 200]
        return pool[:n]

    (cache / "cells_l1.json").write_text(
        json.dumps(
            {
                k: {"examples": sample(v, 6), "n": len(v), "kw": labels.get(k, {}).get("en", "")}
                for k, v in members1.items()
            },
            ensure_ascii=False,
        )
    )
    (cache / "cells_l2.json").write_text(
        json.dumps(
            {
                k: {"examples": sample(v, 5), "n": len(v), "kw": labels.get(k, {}).get("en", "")}
                for k, v in members2.items()
                if len(v) >= 4
            },
            ensure_ascii=False,
        )
    )
    (release_dir / "labels.json").write_text(
        json.dumps(dict(labels), ensure_ascii=False, sort_keys=True, indent=0), encoding="utf-8"
    )
    (release_dir / "examples.json").write_text(
        json.dumps(examples, ensure_ascii=False, sort_keys=True, indent=0), encoding="utf-8"
    )
    counts = {lg: sum(1 for v in labels.values() if lg in v) for lg in langs}
    log(f"labels written: {len(labels)} keys; per language {counts}")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="small")
    ap.add_argument("--release", default=str(ROOT / "src" / "sphota" / "data" / "v1"))
    a = ap.parse_args()
    build_labels(a.scale, Path(a.release))
    from sphota.build.release import refresh_manifest

    refresh_manifest(Path(a.release))


if __name__ == "__main__":
    main()
