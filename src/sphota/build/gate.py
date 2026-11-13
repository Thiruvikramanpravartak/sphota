# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""D-017 decision-gate evaluation and the CPU experiments around it (L1 sweep, anchor check,
v2-mini build).

    python -m sphota.build.gate sweep   --scale small        # L1 size sweep on v1 canonical vectors
    python -m sphota.build.gate anchors --scale small --k K  # e5-small vs LaBSE at L1 size K
    python -m sphota.build.gate v2mini  --scale small --k K  # diversified-sample rebuild + gate report

All metrics use FLORES-200 devtest (held out). Only the top level (L1) is gated.
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np

from sphota.build import embed
from sphota.build.common import ROOT, log
from sphota.build.fit import fit_space, kmeans, l2n

GATE = {"coverage": 0.60, "translation_confident": 0.80, "cross_model": 0.60, "collapse_max": 0.05}
ANCHOR = "intfloat/multilingual-e5-small"
ADAPTER_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "thenlper/gte-small",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/LaBSE",
]


# ------------------------------------------------------------------ L1 quantizer with confidence


def d2top(Y: np.ndarray, C: np.ndarray, m: int = 16) -> tuple[np.ndarray, np.ndarray]:
    Y = Y.astype(np.float64)
    C = C.astype(np.float64)
    d2 = (C**2).sum(1)[None] - 2 * Y @ C.T + (Y**2).sum(1, keepdims=True)
    m = min(m, C.shape[0])
    part = np.argpartition(d2, m - 1, 1)[:, :m]
    dd = np.take_along_axis(d2, part, 1)
    o = np.argsort(dd, 1)
    return np.take_along_axis(part, o, 1), np.take_along_axis(dd, o, 1)


def calibrate_T(C: np.ndarray, Ya: np.ndarray, Yb: np.ndarray) -> float:
    """Max-likelihood temperature: predict the partner's L1 cell (same as release.py, L1 only)."""
    ca, da = d2top(Ya, C)
    cb, db = d2top(Yb, C)
    rows = [(da - da[:, :1], ca == cb[:, :1]), (db - db[:, :1], cb == ca[:, :1])]
    span = float(np.median(rows[0][0][:, 1])) + 1e-9
    best, best_ll = span, -np.inf
    for T in span * np.logspace(-2, 1.5, 50):
        ll = 0.0
        for dd, hit in rows:
            lg = -dd / T
            lg -= lg.max(1, keepdims=True)
            lp = lg - np.log(np.exp(lg).sum(1, keepdims=True))
            ll += float(np.where(hit.any(1), (lp * hit).sum(1), np.log(1e-4)).sum())
        if ll > best_ll:
            best, best_ll = float(T), ll
    return best


def assign(Y: np.ndarray, C: np.ndarray, T: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cand, dd = d2top(Y, C)
    p = np.exp(-(dd - dd[:, :1]) / T)
    p /= p.sum(1, keepdims=True)
    return cand[:, 0], p[:, 0], cand


# ------------------------------------------------------------------ gate metrics


def gate_metrics(
    C, T, Ytest: dict, Ytrain: np.ndarray, train_langs: np.ndarray, Yadapted: dict | None = None
) -> dict:
    """Ytest: lang -> (n, D) held-out canonical vectors (row i = same meaning across languages)."""
    langs = list(Ytest)
    A = {lg: assign(Ytest[lg], C, T) for lg in langs}
    conf_all = np.concatenate([A[lg][1] for lg in langs])
    coverage = float((conf_all >= 0.7).mean())
    en_i, en_p, en_c = A["en"]
    m = en_p >= 0.7
    agree = np.mean([(en_i == A[lg][0]).mean() for lg in langs if lg != "en"])
    agree_conf = (
        np.mean([(en_i[m] == A[lg][0][m]).mean() for lg in langs if lg != "en"])
        if m.any()
        else float("nan")
    )
    top2 = np.mean(
        [((A[lg][2][:, :2] == en_i[:, None]).any(1)).mean() for lg in langs if lg != "en"]
    )
    # language collapse: L1 cells whose training members are >= 80% one language
    tr_cells = assign(Ytrain, C, T)[0]
    dom: dict[int, str] = {}
    by = collections.defaultdict(list)
    for c, lg in zip(tr_cells, train_langs):
        by[int(c)].append(lg)
    for c, v in by.items():
        lg, n = collections.Counter(v).most_common(1)[0]
        if len(v) >= 5 and n / len(v) >= 0.8 and lg != "en":
            dom[c] = lg
    collapse = {
        lg: float(np.mean([dom.get(int(c)) == lg for c in A[lg][0]])) for lg in langs if lg != "en"
    }
    out = {
        "K1": int(C.shape[0]),
        "T": T,
        "coverage": round(coverage, 4),
        "coverage_per_lang": {lg: round(float((A[lg][1] >= 0.7).mean()), 4) for lg in langs},
        "translation_all": round(float(agree), 4),
        "translation_confident": round(float(agree_conf), 4),
        "confident_en_share": round(float(m.mean()), 4),
        "translation_top2": round(float(top2), 4),
        "collapse_max": round(max(collapse.values()), 4),
        "collapse_worst_lang": max(collapse, key=collapse.get),
        "collapse_per_lang": {k: round(v, 4) for k, v in collapse.items()},
        "used_cells_frac": round(float(len(np.unique(tr_cells)) / C.shape[0]), 4),
    }
    if Yadapted:
        xm = {mdl: float((assign(Y, C, T)[0] == en_i).mean()) for mdl, Y in Yadapted.items()}
        out["cross_model"] = round(float(np.mean(list(xm.values()))), 4)
        out["cross_model_per_model"] = {k.split("/")[-1]: round(v, 4) for k, v in xm.items()}
    return out


def verdict(m: dict) -> dict:
    return {
        "1_coverage": (m["coverage"], m["coverage"] >= GATE["coverage"]),
        "2_translation_confident": (
            m["translation_confident"],
            m["translation_confident"] >= GATE["translation_confident"],
        ),
        "3_cross_model": (m.get("cross_model"), (m.get("cross_model") or 0) >= GATE["cross_model"]),
        "4_collapse_max": (m["collapse_max"], m["collapse_max"] <= GATE["collapse_max"]),
    }


# ------------------------------------------------------------------ data helpers


def v1_inputs(scale: str):
    from sphota.build.release import SPLIT_MEANING, embeddings, load, tatoeba_split
    from sphota.release import load_release
    from sphota.space import project

    rel = load_release()
    langs, dev, test, wiki, tat = load(scale)
    E = embeddings(scale)
    fit_ids = np.arange(SPLIT_MEANING)
    cal_ids = np.arange(SPLIT_MEANING, len(dev))
    _, ca = tatoeba_split(len(tat))
    P = lambda X: project(rel, X)
    corpus = np.concatenate([E["wiki"]] + [E[("dev", lg)][fit_ids] for lg in langs])
    corpus_langs = np.concatenate([wiki["lang"].values] + [[lg] * len(fit_ids) for lg in langs])
    Ya = np.concatenate([P(E[("dev", "en")][cal_ids])] * (len(langs) - 1) + [P(E["tat_en"][ca])])
    Yb = np.concatenate([P(E[("dev", lg)][cal_ids]) for lg in langs[1:]] + [P(E["tat_x"][ca])])
    Ytest = {lg: P(E[("test", lg)]) for lg in langs}
    return rel, langs, E, wiki, P, corpus, corpus_langs, Ya, Yb, Ytest


def adapted_en(scale: str, test_en: list[str]) -> dict:
    from sphota.adapters import get_adapter

    out = {}
    for m in ADAPTER_MODELS:
        try:
            out[m] = get_adapter(m)(embed.cached(m, "flores_devtest_en", test_en, scale))
        except Exception as e:  # pragma: no cover
            log(f"skip {m}: {e}")
    return out


# ------------------------------------------------------------------ experiments
# (main() is at the end of the file)


def sweep(scale: str, sizes=(64, 256, 1024, 4096)) -> list[dict]:
    rel, langs, E, wiki, P, corpus, corpus_langs, Ya, Yb, Ytest = v1_inputs(scale)
    from sphota.build.release import load

    test = load(scale)[2]
    Yad = adapted_en(scale, test["en"].tolist())
    Yc = P(corpus)
    res = []
    for k in sizes:
        t0 = time.time()
        C = rel.codebooks[0] if k == rel.sizes[0] else kmeans(Yc, k, seed=0, iters=30)
        T = calibrate_T(C, Ya, Yb)
        m = gate_metrics(C, T, Ytest, Yc, corpus_langs, Yad)
        m["seconds"] = round(time.time() - t0, 1)
        m["note"] = "v1 codebook" if k == rel.sizes[0] else "re-clustered on v1 canonical vectors"
        log(
            f"K1={k}: cov {m['coverage']} tr {m['translation_all']} tr|conf {m['translation_confident']} "
            f"xm {m['cross_model']} collapse {m['collapse_max']} top2 {m['translation_top2']}"
        )
        res.append(m)
    return res


def anchor_check(
    scale: str,
    sizes=(32, 64),
    anchors=(ANCHOR, "sentence-transformers/LaBSE"),
    panel=("sentence-transformers/all-MiniLM-L6-v2",),
) -> list[dict]:
    """Same corpus (20k shared texts + FLORES dev[:800], 12 langs), same meaning-LDA, same L1 sizes.

    Cross-model agreement is measured with ridge adapters (fitted on the same 20k texts) from
    the registry models into each anchor's space."""
    from sphota.build.adapters import training_texts
    from sphota.build.release import load
    from sphota.certify import fit_adapter

    langs, dev, test, wiki, tat = load(scale)
    texts = training_texts(scale, 20000)
    n_fit = 800
    out = []
    for anchor in anchors:
        Xtr = embed.cached(anchor, f"adapter-train-{len(texts)}", texts, scale)
        dev_fit = {
            lg: embed.cached(anchor, f"flores_dev_{lg}", dev[lg].tolist()[:n_fit], scale)
            for lg in langs
        }
        groups = [np.stack([dev_fit[lg][i] for lg in langs]) for i in range(n_fit)]
        corpus = np.concatenate([Xtr] + [dev_fit[lg] for lg in langs])
        corpus_langs = np.array(
            ["mix"] * len(Xtr) + [lg for lg in langs for _ in range(n_fit)], dtype=object
        )
        sp = fit_space(corpus, groups, D=256, P=min(384, corpus.shape[1]), reg=0.05)
        Ytest = {
            lg: sp(embed.cached(anchor, f"flores_devtest_{lg}", test[lg].tolist(), scale))
            for lg in langs
        }
        # calibration: second half of each language's dev fit rows is not held out -> use devtest-free split
        cal = slice(600, 800)
        Ya = np.concatenate([sp(dev_fit["en"][cal])] * (len(langs) - 1))
        Yb = np.concatenate([sp(dev_fit[lg][cal]) for lg in langs[1:]])
        Ytr = sp(Xtr)
        Yad = {}
        # CPU budget: cross-model panel = the other anchor(s) + a small fixed panel
        for m in [x for x in (*anchors, *panel) if x != anchor]:
            Xm = embed.cached(m, f"adapter-train-{len(texts)}", texts, scale)
            Yad[m] = fit_adapter(m, Xm, Ytr)(
                embed.cached(m, "flores_devtest_en", test["en"].tolist(), scale)
            )
        Yc = sp(corpus)
        for k in sizes:
            C = kmeans(Yc, k, seed=0, iters=30)
            T = calibrate_T(C, Ya, Yb)
            m = gate_metrics(C, T, Ytest, Yc, corpus_langs, Yad)
            m["anchor"] = anchor
            log(
                f"{anchor} K1={k}: cov {m['coverage']} tr|conf {m['translation_confident']} xm {m['cross_model']} collapse {m['collapse_max']}"
            )
            out.append(m)
    return out


def diversify(
    wiki,
    X: np.ndarray,
    per_category: int = 40,
    max_per_lang: int = 8000,
    dup_cos: float = 0.95,
    seed: int = 0,
) -> np.ndarray:
    """Indices of a more varied sample: cap sentences per title category (first title word)
    and per language, and drop near-duplicates (cosine > dup_cos) within each language."""
    rng = np.random.default_rng(seed)
    keep = []
    for lg in sorted(set(wiki["lang"])):
        idx = np.where(wiki["lang"].values == lg)[0]
        rng.shuffle(idx)
        cat = wiki["title"].values[idx]
        cat = np.array([str(t).split()[0].lower() if str(t).split() else "" for t in cat])
        counts: collections.Counter = collections.Counter()
        chosen = []
        for i, c in zip(idx, cat):
            if counts[c] >= per_category:
                continue
            counts[c] += 1
            chosen.append(i)
        chosen = np.array(chosen)
        V = l2n(X[chosen].astype(np.float32))
        alive = np.ones(len(chosen), bool)
        for s in range(0, len(chosen), 2048):
            S = V[s : s + 2048] @ V.T
            for r in range(S.shape[0]):
                i = s + r
                if not alive[i]:
                    continue
                dup = np.where(S[r, i + 1 :] > dup_cos)[0] + i + 1
                alive[dup] = False
        chosen = chosen[alive][:max_per_lang]
        keep.append(chosen)
        log(f"diversify {lg}: {len(idx)} -> {len(chosen)}")
    return np.sort(np.concatenate(keep))


MASSIVE_CODE = {"zh": "zh-CN"}  # FLORES-style code -> MASSIVE locale code


def massive_parallel(scale: str, langs: list[str], anchor: str, split: str = "train") -> dict:
    """Anchor embeddings of MASSIVE ``split`` utterances aligned across ``langs`` (same id in
    every language). Train items whose English text also occurs in the test split are dropped."""
    import gzip

    from sphota.build.domains import _hf

    def rows(lg: str, sp: str) -> dict:
        with gzip.open(
            _hf("mteb/amazon_massive_intent", f"{sp}/{MASSIVE_CODE.get(lg, lg)}.json.gz"), "rt"
        ) as f:
            return {r["id"]: r["text"] for r in (json.loads(line) for line in f)}

    R = {lg: rows(lg, split) for lg in langs}
    ids = sorted(set.intersection(*(set(r) for r in R.values())), key=int)
    if split == "train":
        test_en = set(rows("en", "test").values())
        ids = [i for i in ids if R["en"][i] not in test_en]
    if scale == "tiny":
        ids = ids[:1500]
    E = {
        lg: embed.cached(anchor, f"massive-{split}-{lg}-{len(ids)}", [R[lg][i] for i in ids], scale)
        for lg in langs
    }
    return {"ids": ids, "E": E}


def massive_xling(sp, C: np.ndarray, scale: str, langs: list[str], anchor: str) -> dict:
    """Held-out MASSIVE test commands: share in the same top-level cell as the English version."""
    from sphota.build.domains import nearest

    m = massive_parallel(scale, langs, anchor, split="test")
    cell = {lg: nearest(sp(m["E"][lg]), C) for lg in langs}
    per = {lg: round(float((cell[lg] == cell["en"]).mean()), 4) for lg in langs if lg != "en"}
    return {"mean": round(float(np.mean(list(per.values()))), 4), "per_lang": per}


def v2mini(
    scale: str,
    sizes=(32, 64),
    out_dir: Path | None = None,
    anchor: str = ANCHOR,
    adapter_models: list[str] | None = None,
    n_adapter: int = 40000,
    adapter_kind: str = "ridge",
    extra_parallel: str | None = None,
    seed: int = 0,
    extra_n: int | None = None,
) -> dict:
    """Rebuild the space on a diversified sample, evaluate the D-017 gate, keep the best build
    as an experimental release (never overwrites frozen v1).

    ``adapter_kind``: ``"ridge"`` (as in v1) or ``"mlp"`` (v1.1-style nonlinear, D-028).
    ``extra_parallel``: ``"massive"`` adds MASSIVE train commands (aligned across the 12
    languages, test texts excluded) to the translation scatter of meaning-LDA, so the space also
    learns that short informal commands in different languages are the same (D-035)."""
    from sphota.build.adapters import training_texts
    from sphota.build.label_filter import filter_labels
    from sphota.build.labels import ctfidf
    from sphota.build.release import SPLIT_MEANING, embeddings, load, tatoeba_split
    from sphota.certify import fit_adapter

    langs, dev, test, wiki, tat = load(scale)
    E = embeddings(scale, model=anchor)
    from sphota.build.common import SCALES

    keep = diversify(wiki, E["wiki"], max_per_lang=SCALES[scale].get("max_per_lang", 8000))
    fit_ids = np.arange(SPLIT_MEANING)
    cal_ids = np.arange(SPLIT_MEANING, len(dev))
    tr, ca = tatoeba_split(len(tat))
    groups = [np.stack([E[("dev", lg)][i] for lg in langs]) for i in fit_ids]
    groups += [np.stack([E["tat_x"][i], E["tat_en"][i]]) for i in tr]
    if extra_parallel == "massive":
        mp = massive_parallel(scale, langs, anchor)
        idx = range(len(mp["ids"]))
        if extra_n is not None and extra_n < len(mp["ids"]):  # exploratory balance (D-043)
            idx = np.sort(
                np.random.default_rng(seed).choice(len(mp["ids"]), extra_n, replace=False)
            )
        groups += [np.stack([mp["E"][lg][i] for lg in langs]) for i in idx]
        log(f"meaning-LDA: + {len(idx)} MASSIVE train meanings x {len(langs)} languages")
    corpus = np.concatenate([E["wiki"][keep]] + [E[("dev", lg)][fit_ids] for lg in langs])
    corpus_langs = np.concatenate(
        [wiki["lang"].values[keep]] + [[lg] * len(fit_ids) for lg in langs]
    )
    sp = fit_space(corpus, groups, D=256, P=384, reg=0.05)
    Yc = sp(corpus)
    Ya = np.concatenate([sp(E[("dev", "en")][cal_ids])] * (len(langs) - 1) + [sp(E["tat_en"][ca])])
    Yb = np.concatenate([sp(E[("dev", lg)][cal_ids]) for lg in langs[1:]] + [sp(E["tat_x"][ca])])
    Ytest = {lg: sp(E[("test", lg)]) for lg in langs}
    texts = training_texts(scale, n_adapter)
    Ytr = sp(embed.cached(anchor, f"adapter-train-{len(texts)}", texts, scale))
    Yad = {}
    for m in [x for x in (adapter_models or ADAPTER_MODELS) if x != anchor]:
        # use the adapter training set already embedded for this model (LaBSE: 20k, others: 40k)
        n_m = 20000 if "LaBSE" in m and n_adapter > 20000 else n_adapter
        texts_m = texts if n_m == len(texts) else training_texts(scale, n_m)
        Xm = embed.cached(m, f"adapter-train-{len(texts_m)}", texts_m, scale)
        Ytr_m = (
            Ytr
            if n_m == len(texts)
            else sp(embed.cached(anchor, f"adapter-train-{len(texts_m)}", texts_m, scale))
        )
        if adapter_kind == "mlp":
            from sphota.build.adapters_v11 import fit_mlp2

            ad_m = fit_mlp2(m, Xm, Ytr_m)
        else:
            ad_m = fit_adapter(m, Xm, Ytr_m)
        Yad[m] = ad_m(embed.cached(m, "flores_devtest_en", test["en"].tolist(), scale))
    results = []
    best = None
    for k in sizes:
        C = kmeans(Yc, k, seed=seed, iters=40)
        T = calibrate_T(C, Ya, Yb)
        m = gate_metrics(C, T, Ytest, Yc, corpus_langs, Yad)
        m["seed"] = seed
        m["sample"] = {"wiki_sentences": len(keep), "of": len(wiki)}
        m["adapter_kind"] = adapter_kind
        m["extra_parallel"] = extra_parallel
        try:
            m["massive_xling_same_cell"] = massive_xling(sp, C, scale, langs, anchor)
        except Exception as e:  # pragma: no cover - network dependent
            log(f"massive cross-lingual eval skipped ({type(e).__name__})")
        m["verdict"] = {k2: {"value": v[0], "pass": bool(v[1])} for k2, v in verdict(m).items()}
        log(
            f"v2-mini K1={k}: "
            + ", ".join(
                f"{k2}={v['value']} {'PASS' if v['pass'] else 'FAIL'}"
                for k2, v in m["verdict"].items()
            )
        )
        results.append(m)
        score = (sum(v["pass"] for v in m["verdict"].values()), k)
        if best is None or score > best[0]:
            best = (score, k, C, T)
    # experimental release: L1 = best size, L2 = 256 residual cells, with c-TF-IDF labels + filter
    _, k, C, T = best
    l1 = assign(Yc, C, T)[0]
    R = Yc - C[l1]
    C2 = kmeans(R, 256, seed=1, iters=30)
    out_dir = out_dir or ROOT / "experiments" / "v2-mini"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "space.npz", mu=sp.mu, W=sp.W)
    np.savez(out_dir / "codebooks.npz", level1=C.astype(np.float32), level2=C2.astype(np.float32))
    members: dict[str, list] = collections.defaultdict(list)
    texts_all = list(wiki["text"].values[keep]) + [
        t for lg in langs for t in dev[lg].tolist()[: len(fit_ids)]
    ]
    srcs = [f"w:{t}" for t in wiki["title"].values[keep]] + [
        f"f:{i}" for lg in langs for i in fit_ids
    ]
    for c, lg, t, s in zip(l1, corpus_langs, texts_all, srcs):
        members[str(int(c))].append((lg, t, s, t))
    labels: dict[str, dict[str, str]] = collections.defaultdict(dict)
    for lg in langs:
        for key, lab in ctfidf(members, lg, None, top=3, min_sources=2).items():
            labels[key][lg] = lab
    kept, review = filter_labels(dict(labels))
    (out_dir / "labels.json").write_text(
        json.dumps(kept, ensure_ascii=False, indent=0), encoding="utf-8"
    )
    meta = {
        "status": "EXPERIMENTAL — not a frozen space; do not use for stored codes",
        "built_from": f"{scale} scale, diversified per-language Wikipedia sample",
        "anchor": anchor,
        "sizes": [int(k), 256],
        "temperature_l1": T,
        "label_filter": {"kept_cells": len(kept), "review_cells": len(review)},
        "gate": [r["verdict"] | {"K1": r["K1"]} for r in results],
    }
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {
        "results": results,
        "kept_K1": int(k),
        "labels_kept": len(kept),
        "labels_review": len(review),
    }


def print_report(res: dict) -> None:
    """D-017 PASS/FAIL table for the kept build."""
    r = next(x for x in res["results"] if x["K1"] == res["kept_K1"])
    print(f"\nD-017 decision gate (K1={r['K1']}, held-out FLORES devtest):")
    names = {
        "1_coverage": "coverage >= 60%",
        "2_translation_confident": "translation (confident) >= 80%",
        "3_cross_model": "cross-model >= 60%",
        "4_collapse_max": "language collapse <= 5%",
    }
    for k, v in r["verdict"].items():
        val = "n/a" if v["value"] is None else f"{100 * v['value']:.1f}%"
        print(f"  {names[k]:<34} {val:>7}  {'PASS' if v['pass'] else 'FAIL'}")
    allp = all(v["pass"] for v in r["verdict"].values())
    if r["K1"] < 256:
        print(
            f"  OVERALL: NOT A VALID GATE RESULT - only {r['K1']} regions (D-018 requires >= 256)"
        )
    else:
        print(
            f"  OVERALL: {'PASS - fine codes hold across languages and models' if allp else 'FAIL - keep the regions + migration positioning (D-017)'}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sweep", "anchors", "v2mini"])
    ap.add_argument("--scale", default="small")
    ap.add_argument("--anchor", default=ANCHOR)
    ap.add_argument("--sizes", type=int, nargs="*", default=[32, 64])
    ap.add_argument("--adapter-models", nargs="*", default=None)
    ap.add_argument("--n-adapter", type=int, default=40000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--adapter-kind", choices=["ridge", "mlp"], default="ridge")
    ap.add_argument("--extra-parallel", choices=["massive"], default=None)
    ap.add_argument("--seed", type=int, default=0, help="k-means seed (confirmation runs, D-035)")
    ap.add_argument(
        "--extra-n", type=int, default=None, help="cap on extra parallel meanings (D-043)"
    )
    a = ap.parse_args()
    if a.cmd == "sweep":
        res = sweep(a.scale)
        (ROOT / "bench" / "results" / "l1-sweep.json").write_text(
            json.dumps(res, indent=1), encoding="utf-8"
        )
    elif a.cmd == "anchors":
        res = anchor_check(a.scale)
        (ROOT / "bench" / "results" / "anchor-check.json").write_text(
            json.dumps(res, indent=1), encoding="utf-8"
        )
    elif a.cmd == "v2mini":
        out = Path(a.out) if a.out else None
        res = v2mini(
            a.scale,
            sizes=tuple(a.sizes),
            out_dir=out,
            anchor=a.anchor,
            adapter_models=a.adapter_models,
            n_adapter=a.n_adapter,
            adapter_kind=a.adapter_kind,
            extra_parallel=a.extra_parallel,
            seed=a.seed,
            extra_n=a.extra_n,
        )
        suffix = (
            ("-mlp" if a.adapter_kind == "mlp" else "")
            + (f"-{a.extra_parallel}" if a.extra_parallel else "")
            + (f"-seed{a.seed}" if a.seed else "")
            + (f"-n{a.extra_n}" if a.extra_n else "")
        )
        name = (
            "v2-mini-gate.json"
            if a.scale == "small" and a.anchor == ANCHOR and not suffix
            else f"v2-gate-{a.scale}-{embed.short(a.anchor)}{suffix}.json"
        )
        (ROOT / "bench" / "results" / name).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print_report(res)


if __name__ == "__main__":
    main()
