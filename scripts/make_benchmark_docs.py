"""Render docs/benchmark.md and bench/leaderboard/index.html from bench/results/*.json.

python scripts/make_benchmark_docs.py
"""

from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "bench" / "results"


def load(name: str) -> dict:
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def row(vals: list[float]) -> str:
    return " | ".join(pct(v) for v in vals)


def main() -> None:
    b = load("benchmark-small.json")
    ad = load("adapters-small.json")
    v2v = load("vec2vec-small.json")
    sweep = load("design-sweep-small.json")
    langs = b["languages"]
    xl = b["cross_lingual"]
    md = [
        "# Benchmarks",
        "",
        "All numbers are measured on data **never used** to fit the space, codebooks,",
        "calibration or adapters: FLORES-200 *devtest* (1012 sentences × 12 languages), SIB-200",
        "topic labels for those sentences, and STS-B paraphrase pairs scored ≥ 4.5/5.",
        f"Space `{b['space']}`, built at `small` scale on CPU. Raw results:",
        "[`bench/results/`](https://github.com/thiruvikramanpravartak/sphota/tree/main/bench/results).",
        "Reproduce: `python -m sphota.build.bench --scale small`.",
        "",
        '!!! note "How to read agreement"',
        "    *Agreement at L2* = fraction of pairs whose codes share the first two levels. A random",
        "    pair shares L1 with probability ≈ 1/4096. Discretizing a continuous space always",
        "    splits some equivalent texts across a boundary, so agreement falls with depth;",
        "    confidence (below) tells you which codes to trust.",
        "",
        "## 1. Cross-lingual agreement (anchor)",
        "",
        "English sentence vs. its human translation, same code prefix:",
        "",
        "| | L1 | L2 | L3 | L4 |",
        "|---|---|---|---|---|",
        f"| **mean over 11 languages** | {row(xl['mean'])} |",
        f"| all language pairs | {row(xl['all_pairs_mean'])} |",
        f"| stable mode | {row(xl['stable_mode_mean'])} |",
    ]
    for lg in langs[1:]:
        md.append(f"| en ↔ {lg} | {row(xl['per_language'][lg])} |")
    if sweep:
        pca = next((r for r in sweep if r["align"] == "pca" and r["K1"] == 4096), None)
        al = next(
            (r for r in sweep if r["align"] == "aligned" and r["K1"] == 4096 and r["D"] == 256),
            None,
        )
        if pca and al:
            md += [
                "",
                f"Effect of cross-lingual alignment (design sweep, same sizes): L1 {pct(pca['agree'][0])} with plain",
                f"PCA-whitening → {pct(al['agree'][0])} with meaning-LDA; L2 {pct(pca['agree'][1])} → {pct(al['agree'][1])}.",
            ]
    para = b["paraphrase"]
    md += [
        "",
        "## 2. Paraphrase stability (English, STS-B ≥ 4.5)",
        "",
        f"{para['n_pairs']} pairs.",
        "",
        "| | L1 | L2 | L3 | L4 |",
        "|---|---|---|---|---|",
        f"| default | {row(para['agree'])} |",
        f"| stable mode | {row(para['stable_mode'])} |",
    ]
    md += [
        "",
        "## 3. Precision truncation (GPS accuracy)",
        "",
        "With `min_confidence=τ`, codes are cut where the cumulative confidence drops below τ.",
        "*Consistent* = the shorter code of the pair is a prefix of the longer one (they do not",
        "contradict each other).",
        "",
        "| τ | consistent (cross-lingual) | mean depth | consistent (paraphrase) | mean depth |",
        "|---|---|---|---|---|",
    ]
    for a, p in zip(b["truncation_cross_lingual"], para["truncation"]):
        md.append(
            f"| {a['min_confidence']} | {pct(a['consistent'])} | {a['mean_depth']:.2f} | {pct(p['consistent'])} | {p['mean_depth']:.2f} |"
        )
    md += [
        "",
        "### Calibration on unseen data",
        "",
        "| level | ECE | bins (confidence → observed agreement) |",
        "|---|---|---|",
    ]
    for r in b["calibration_unseen"]:
        bins = "; ".join(f"{x['conf']:.2f}→{x['agree']:.2f} (n={x['n']})" for x in r["bins"])
        md.append(f"| L{r['level']} | {r['ece']:.3f} | {bins} |")
    u = b["utilization"]
    md += [
        "",
        "## 4. Utilization (no codebook collapse)",
        "",
        "| set | n | L1 codes used | L1 perplexity (of 4096) | largest cell share | L2 / L3 / L4 symbols used |",
        "|---|---|---|---|---|---|",
    ]
    for k, v in u.items():
        md.append(
            f"| {k.replace('_', ' ')} | {v['n']:,} | {v['l1_used']} ({pct(v['l1_used_frac'])}) | {v['l1_perplexity']} | {pct(v['max_cell_share'])} | "
            f"{pct(v['l2_used_frac'])} / {pct(v['l3_used_frac'])} / {pct(v['l4_used_frac'])} |"
        )
    rt = b["retrieval_translation"]
    md += [
        "",
        "## 5. Retrieval with codes vs. raw vectors",
        "",
        "### Translation retrieval (English query → find its translation among 1012, mean of 11 languages)",
        "",
        "| representation | top-1 | recall@10 |",
        "|---|---|---|",
    ]
    names = {
        "raw_anchor": "raw anchor vectors (384-d floats)",
        "canonical": "canonical vectors (256-d floats)",
        "codes": "**codes only** (13 characters)",
        "codes_multiprobe": "codes + L1 alternates (multi-probe)",
    }
    for k, v in rt.items():
        md.append(f"| {names[k]} | {pct(v['top1'])} | {pct(v['recall@10'])} |")
    tp = b["retrieval_topic_sib200"]
    md += [
        "",
        f"### Topic retrieval (SIB-200, 7 topics, n={tp['n']}): precision@10 of same-topic neighbours",
        "",
        "| representation | P@10 |",
        "|---|---|",
    ]
    tnames = {
        "raw_anchor": "raw anchor vectors",
        "canonical": "canonical vectors",
        "codes": "codes only",
        "codes_en_to_hi": "codes, English queries → Hindi documents",
        "raw_anchor_en_to_hi": "raw vectors, English → Hindi",
    }
    for k, v in tp.items():
        if k != "n":
            md.append(f"| {tnames.get(k, k)} | {pct(v['p@10'])} |")
    md += [
        "",
        "Codes are a coarse index: they win on size, readability, and portability, and lose fine",
        "ranking precision. Use them to filter/organize, and re-rank with vectors when you have them.",
    ]
    if ad:
        md += [
            "",
            "## 6. Cross-model agreement (adapters)",
            "",
            "Same text, model + adapter vs. the anchor (FLORES devtest). *Reference* models were used",
            "when fitting the space (their disagreement is treated as noise); held-out models were not.",
            "",
            "| model | role | L1 (en) | L2 (en) | L1 (all langs) | cross-lingual L1 (model) | fidelity (cos) | grade |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for m, v in sorted(ad.items(), key=lambda kv: -kv[1]["agreement_same_text_en"][0]):
            role = "reference" if v.get("reference_model") else "not used to fit the space"
            md.append(
                f"| `{m}` | {role} | {pct(v['agreement_same_text_en'][0])} | {pct(v['agreement_same_text_en'][1])} | "
                f"{pct(v['agreement_same_text'][0])} | {pct(v['agreement_cross_lingual'][0])} | {v['fidelity_en']:.3f} | {v['grade']} |"
            )
    if b.get("int8_variant_agreement"):
        md += [
            "",
            f"Anchor int8 ONNX variant vs fp32 reference (same text): {row(b['int8_variant_agreement'])} (L1..L4).",
        ]
    if v2v:
        u_, s_ = v2v["unsupervised"], v2v["supervised_ridge_reference"]
        md += [
            "",
            "## 8. Unsupervised adapter (vec2vec-style, no paired data)",
            "",
            f"`{v2v['model']}` → s1 with disjoint text sets ({v2v['unpaired_texts_per_side']:,} per side), {v2v['steps']} steps on CPU.",
            "",
            "| | top-1 retrieval (of 1012) | mean cosine | L1 agreement | L2 agreement |",
            "|---|---|---|---|---|",
            f"| unsupervised | {pct(u_['top1_retrieval'])} | {u_['mean_cosine']:.3f} | {pct(u_['agreement'][0])} | {pct(u_['agreement'][1])} |",
            f"| supervised ridge (same budget) | {pct(s_['top1_retrieval'])} | {s_['mean_cosine']:.3f} | {pct(s_['agreement'][0])} | {pct(s_['agreement'][1])} |",
            f"| chance | {pct(v2v['chance_top1'])} | 0 | ≈0 | ≈0 |",
        ]
    sp = b["speed"]
    md += [
        "",
        "## 9. Speed (CPU)",
        "",
        f"* Encode, ONNX fp32: **{sp['encode_fp32_sent_per_s']} sentences/s**; int8: {sp['encode_int8_sent_per_s']} sentences/s ({sp['cpu']}).",
        f"* Quantize precomputed vectors (numpy): {sp['quantize_vectors_per_s']:,} vectors/s.",
        "",
        "## Limitations",
        "",
        "* `small`-scale build: ~78k sentences in 12 languages fit 4096 L1 cells (~19 per cell). Regions for",
        "  rare topics are coarse; a GPU `full`-scale rebuild (s2) is planned.",
        "* The anchor (multilingual-e5-small) is small; very short or idiomatic sentences in some languages",
        "  (e.g. 4-word Hindi or French sentences) get low confidence and sometimes different L1 regions.",
        "* Cross-model agreement at the *exact code* level is modest. For model migration rely on prefix",
        "  search plus alternates, and expect recall rather than identity.",
        "* Paraphrase data is English only (STS-B). Cross-lingual numbers use FLORES (news/wiki style).",
        "* Labels are automatic (LLM names + machine translation) and can be wrong.",
        "",
    ]
    (ROOT / "docs" / "benchmark.md").write_text("\n".join(md), encoding="utf-8")

    # leaderboard
    rows = []
    for m, v in sorted(ad.items(), key=lambda kv: -kv[1]["agreement_same_text_en"][0]):
        rows.append(
            f"<tr><td><code>{html.escape(m)}</code></td><td>adapter</td>"
            + "".join(f"<td>{100 * x:.1f}</td>" for x in v["agreement_same_text_en"][:2])
            + f"<td>{100 * v['agreement_same_text'][0]:.1f}</td><td>{100 * v['agreement_cross_lingual'][0]:.1f}</td>"
            f"<td>{v['fidelity_en']:.3f}</td><td class='g {v['grade']}'>{v['grade']}</td></tr>"
        )
    anchor = "<tr class=anchor><td><code>intfloat/multilingual-e5-small</code> (anchor)</td><td>reference</td><td>100</td><td>100</td><td>100</td><td>100</td><td>1.000</td><td class='g certified-multilingual'>anchor</td></tr>"
    page = f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Sphota Leaderboard</title><style>
:root{{--bg:#0b1120;--card:#121a2e;--text:#e6eeff;--muted:#93a4c3;--a:#4dabf7}}
@media (prefers-color-scheme: light){{:root{{--bg:#f6f8fc;--card:#fff;--text:#132036;--muted:#5b6b86}}}}
body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1080px;margin:0 auto;padding:32px 16px}} h1{{margin:0}} p{{color:var(--muted)}}
.wrap{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;background:var(--card);border-radius:12px;overflow:hidden}}
th,td{{padding:10px 12px;text-align:left;border-bottom:1px solid rgba(128,128,128,.2);white-space:nowrap}}
th{{color:var(--muted);font-weight:600;font-size:13px}} td:nth-child(n+3){{text-align:right;font-variant-numeric:tabular-nums}}
code{{color:var(--a)}} tr.anchor{{opacity:.75}} .g{{font-weight:600}} .certified-multilingual{{color:#2fb344}} .certified{{color:#74b816}} .not-certified{{color:#e8590c}}
</style></head><body><main><h1>Sphota s1 adapter leaderboard</h1>
<p>Agreement with the reference anchor on the same text (FLORES-200 devtest), per level, in %. Fidelity = cosine between
adapted and reference canonical vectors. Thresholds: <a href="../../docs/spec/08-conformance.md">spec Part 8</a>.
Generated by <code>scripts/make_benchmark_docs.py</code>.</p>
<div class=wrap><table><thead><tr><th>model</th><th>role</th><th>L1 exact en</th><th>L2 exact en</th><th>L1 nbhd en</th><th>top-3 all</th><th>fidelity</th><th>grade</th></tr></thead>
<tbody>{anchor}{"".join(rows)}</tbody></table></div></main></body></html>"""
    out = ROOT / "bench" / "leaderboard"
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page, encoding="utf-8")
    print("wrote docs/benchmark.md and bench/leaderboard/index.html")


if __name__ == "__main__":
    main()
