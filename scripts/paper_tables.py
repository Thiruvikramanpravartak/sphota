#!/usr/bin/env python3
# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the paper's (and docs/benchmark.md's) result tables from ``bench/results/*.json`` (D-031: every number in the
paper comes from a script). Tables live between markers in docs/paper/sphota.md:

    <!-- table:NAME -->
    ...generated...
    <!-- /table -->

    python scripts/paper_tables.py            # rewrite the tables in place
    python scripts/paper_tables.py --check    # exit 1 if the paper is out of date
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "bench" / "results"
PAPER = ROOT / "docs" / "paper" / "sphota.md"
TARGETS = [PAPER, ROOT / "docs" / "benchmark.md"]


def pct(x: float | None, d: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def load(name: str) -> dict:
    return json.loads((RES / name).read_text(encoding="utf-8"))


def t_gpu() -> str:
    rows = [
        "| Anchor | L1 cells | Coverage (≥ 60%) | Translation, confident (≥ 80%) | Cross-model (≥ 60%) | Worst collapse (≤ 5%) | Pass |",
        "|---|---|---|---|---|---|---|",
    ]
    for f, name in [
        ("v2-gate-medium-me5-small.json", "multilingual-e5-small"),
        ("v2-gate-medium-labse.json", "LaBSE"),
        ("v2-gate-medium-bge-m3.json", "bge-m3"),
    ]:
        for r in load(f)["results"]:
            v = r["verdict"]
            n = sum(x["pass"] for x in v.values())
            rows.append(
                f"| {name} | {r['K1']} | {pct(v['1_coverage']['value'])} | "
                f"{pct(v['2_translation_confident']['value'])} | {pct(v['3_cross_model']['value'])} | "
                f"{pct(v['4_collapse_max']['value'])} | {n}/4 |"
            )
    return "\n".join(rows)


def t_adapters() -> str:
    d = load("adapter-study-small.json")["models"]
    rows = [
        "| Model | Adapter | Same cell (4096) | Same cell (256) | Same region (32) |",
        "|---|---|---|---|---|",
    ]
    for m, v in d.items():
        for k, label in [
            ("procrustes", "rotation"),
            ("ridge", "linear (v1)"),
            ("mlp", "residual MLP"),
            ("mlp+code", "MLP + L1 code loss"),
            ("mlp+hier", "MLP + L1/region loss"),
        ]:
            x = v[k]
            rows.append(
                f"| {m.split('/')[-1]} | {label} | {pct(x['l1_4096']['all'])} | "
                f"{pct(x['cells_256']['all'])} | {pct(x['regions_32']['all'])} |"
            )
    n = len(d)
    for k, label in [("ridge", "linear (v1)"), ("mlp", "residual MLP")]:
        a = [
            sum(v[k][g]["all"] for v in d.values()) / n
            for g in ["l1_4096", "cells_256", "regions_32"]
        ]
        rows.append(
            f"| **mean of {n}** | **{label}** | **{pct(a[0])}** | **{pct(a[1])}** | **{pct(a[2])}** |"
        )
    return "\n".join(rows)


def t_scaling() -> str:
    sc = load("adapter-study-small.json")["scaling"]
    models = list(sc)
    head = (
        "| Training pairs | "
        + " | ".join(f"{m.split('/')[-1]}: linear | MLP" for m in models)
        + " |"
    )
    rows = [head, "|---|" + "---|---|" * len(models)]
    for n in ["2500", "5000", "10000", "20000", "40000"]:
        rows.append(
            f"| {int(n):,} | "
            + " | ".join(
                f"{pct(sc[m][n]['ridge'])} | {pct(sc[m][n].get('mlp+hier', sc[m][n].get('mlp+code')))}"
                for m in models
            )
            + " |"
        )
    return "\n".join(rows)


def t_v11() -> str:
    d = load("adapters-v1.1-small.json")
    rows = [
        "| Model | Same cell, English (v1 → v1.1) | Same cell, 12 languages (v1 → v1.1) | Migration recall@10 (v1 → v1.1) | Grade |",
        "|---|---|---|---|---|",
    ]
    for m, v in d.items():
        a, b = v["v1"], v["v1.1"]
        rows.append(
            f"| {m.split('/')[-1]} | {pct(a['agreement_same_text_en'][0])} → {pct(b['agreement_same_text_en'][0])} | "
            f"{pct(a['agreement_same_text'][0])} → {pct(b['agreement_same_text'][0])} | "
            f"{pct(v['retrieval_r10_en']['v1'])} → {pct(v['retrieval_r10_en']['v1.1'])} | {b['grade']} |"
        )
    return "\n".join(rows)


def t_indic() -> str:
    d = load("indic-small.json")
    rows = [
        "| Language | Region = English | Agreement when both confident | First block = English | Retrieval@1 | Top-region share |",
        "|---|---|---|---|---|---|",
    ]
    e = d["english"]
    rows.append(f"| English (reference) | | | | | {pct(e['top_region_share'])} |")
    for m in d["languages"].values():
        rows.append(
            f"| {m['name']} | {pct(m['region_agree'])} | {pct(m['region_agree_confident'])} | "
            f"{pct(m['l1_agree'])} | {pct(m['retrieval_at_1'])} | {pct(m['top_region_share'])} |"
        )
    return "\n".join(rows)


def _dom() -> dict:
    return load("domains-small.json")["domains"]


DOMAIN_NAMES = {
    "conversational": "Conversational (MASSIVE)",
    "customer_support": "Customer support (Banking77)",
    "technical_qa": "Technical Q&A (Stack Exchange)",
    "science": "Science (arXiv titles)",
    "reviews_indic": "Product reviews (IndicReviews)",
}


def t_domains() -> str:
    d = _dom()
    rows = [
        "| Domain | Lang. | n | Labels | Sphota themes: V / purity (groups) | k-means, same #groups: V / purity | Sphota regions: V | k-means 32: V | k-means true k: V | TF-IDF: V |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, name in DOMAIN_NAMES.items():
        for lg, x in d[key]["grouping"].items():
            t, k = x["sphota_themes"], x["kmeans_same_as_themes"]
            rows.append(
                f"| {name} | {lg} | {x['n']:,} | {x['labels']} | {t['v_measure']:.3f} / {pct(t['purity'], 0)} ({t['groups']}) | "
                f"{k['v_measure']:.3f} / {pct(k['purity'], 0)} | {x['sphota_regions']['v_measure']:.3f} | "
                f"{x['kmeans_32']['v_measure']:.3f} | {x['kmeans_k_true']['v_measure']:.3f} | "
                f"{x['tfidf_kmeans_k_true']['v_measure']:.3f} |"
            )
    return "\n".join(rows)


def t_xling() -> str:
    d = _dom()
    c, r = d["conversational"]["cross_lingual_vs_en"], d["reviews_indic"]["cross_lingual_vs_hi"]
    rows = ["| Data (pivot) | Language | Same theme | Same region |", "|---|---|---|---|"]
    rows += [
        f"| MASSIVE (English) | {lg} | {pct(v['same_theme'])} | {pct(v['same_region'])} |"
        for lg, v in c.items()
    ]
    rows += [
        f"| IndicReviews (Hindi) | {lg} | {pct(v['same_theme'])} | {pct(v['same_region'])} |"
        for lg, v in r.items()
    ]
    return "\n".join(rows)


def t_xmodel() -> str:
    x = _dom()["conversational"]["cross_model_pmMiniLM"]
    rows = [
        "| MASSIVE language | Adapter | Same theme as anchor | Same region as anchor |",
        "|---|---|---|---|",
    ]
    for lg in ["en", "hi"]:
        for s in ["v1", "v1.1"]:
            v = x[f"{lg}_{s}"]
            rows.append(f"| {lg} | {s} | {pct(v['same_theme'])} | {pct(v['same_region'])} |")
    return "\n".join(rows)


def t_code() -> str:
    d = _dom()["code"]["per_language"]
    rows = [
        "| Language | n | Doc and code: same theme | same region | Retrieval R@1 / R@10, raw anchor | canonical space |",
        "|---|---|---|---|---|---|",
    ]
    for lg, v in d.items():
        a, c = v["retrieval_raw_anchor"], v["retrieval_canonical"]
        rows.append(
            f"| {lg} | {v['n']} | {pct(v['same_theme'])} | {pct(v['same_region'])} | "
            f"{pct(a['r@1'])} / {pct(a['r@10'])} | {pct(c['r@1'])} / {pct(c['r@10'])} |"
        )
    return "\n".join(rows)


def t_refine() -> str:
    rows = [
        "| Round | Winner (chosen on dev) | Fresh test data | R1 quality | R2 stable | R3 mixed languages | R4 speed |",
        "|---|---|---|---|---|---|---|",
    ]
    r1 = load("refine-study-small.json")
    w = r1["winner"]
    t = r1["test"][w]
    rows.append(
        f"| 1 | {w} | test splits (22 sets) | {t['R1_passed']}/{t['datasets']} | "
        f"{'pass' if t['R2_pass'] else 'fail'} | {sum(x['pass'] for x in t['R3'].values())}/{len(t['R3'])} | "
        f"{'pass' if r1['R4']['R4_python_pass'] else 'fail'} |"
    )
    fresh = {
        2: "confirmation set 1",
        3: "set 2 (13 new MASSIVE languages)",
        4: "set 3 (24 more languages)",
        5: "set 4 (MASSIVE validation)",
        6: "set 5 (Stack Exchange, arXiv, MasakhaNEWS train, FLORES)",
    }
    for n in range(2, 7):
        d = load(f"refine-round{n}-small.json")
        r3 = d["R3"]
        rows.append(
            f"| {n} | {d['winner']} | {fresh[n]} | {d['R1_passed']}/{d['R1_sets']} | "
            f"{'pass' if d['R2_pass'] else 'fail'} | {sum(x['pass'] for x in r3.values())}/{len(r3)} | "
            f"{'pass' if d['R4']['R4_python_pass'] else 'fail'} |"
        )
    return "\n".join(rows)


def t_shorttext() -> str:
    rows = [
        "| MASSIVE meanings in the scatter | Cells | Short commands, same cell as English | Coverage | Translation, confident | Cross-model | Worst collapse |",
        "|---|---|---|---|---|---|---|",
    ]
    for f, label in [
        ("shorttext-baseline-small.json", "0 (baseline)"),
        ("shorttext-massive-n800-small.json", "800"),
        ("shorttext-massive-n2500-small.json", "2,500"),
        ("shorttext-massive-small.json", "11,492"),
    ]:
        for r in load(f)["results"]:
            rows.append(
                f"| {label} | {r['K1']} | {pct(r['massive_xling_same_cell']['mean'])} | {pct(r['coverage'])} | "
                f"{pct(r['translation_confident'])} | {pct(r.get('cross_model'))} | {pct(r['collapse_max'])} ({r['collapse_worst_lang']}) |"
            )
    return "\n".join(rows)


TABLES = {
    "gpu": t_gpu,
    "adapters": t_adapters,
    "scaling": t_scaling,
    "v11": t_v11,
    "indic": t_indic,
    "domains": t_domains,
    "xling": t_xling,
    "xmodel": t_xmodel,
    "code": t_code,
    "refine": t_refine,
    "shorttext": t_shorttext,
}


def render(text: str) -> str:
    def sub(m: re.Match) -> str:
        name = m.group(1)
        return f"<!-- table:{name} -->\n{TABLES[name]()}\n<!-- /table -->"

    return re.sub(r"<!-- table:(\w+) -->.*?<!-- /table -->", sub, text, flags=re.S)


if __name__ == "__main__":
    stale = []
    for path in TARGETS:
        src = path.read_text(encoding="utf-8")
        out = render(src)
        if out != src:
            stale.append(path.name)
            if "--check" not in sys.argv:
                path.write_text(out, encoding="utf-8")
    if "--check" in sys.argv:
        if stale:
            print(f"tables out of date in {stale}: run python scripts/paper_tables.py")
        sys.exit(1 if stale else 0)
    print(f"tables refreshed in {', '.join(p.name for p in TARGETS)}")
