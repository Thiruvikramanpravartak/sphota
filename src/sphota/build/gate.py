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
    C,
    T,
    Ytest: dict,
    Ytrain: np.ndarray,
    train_langs: np.ndarray,
    Yadapted: dict | None = None,
    tr_cells: np.ndarray | None = None,
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
    if tr_cells is None:  # callers evaluating many adapters on one codebook pass it in
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


def massive_texts(langs: list[str], split: str, scale: str) -> dict[str, list[str]]:
    """MASSIVE ``split`` utterances aligned across ``langs`` (same id in every language), in id
    order. Train items whose English text also occurs in the test split are dropped."""
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
    return {lg: [R[lg][i] for i in ids] for lg in langs}


def massive_parallel(scale: str, langs: list[str], anchor: str, split: str = "train") -> dict:
    """Anchor embeddings of MASSIVE ``split`` utterances aligned across ``langs``."""
    T = massive_texts(langs, split, scale)
    n = len(T["en"])
    E = {lg: embed.cached(anchor, f"massive-{split}-{lg}-{n}", T[lg], scale) for lg in langs}
    return {"ids": list(range(n)), "E": E}


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
        # LaBSE was capped at 20k pairs to save CPU time; on a GPU every model uses n_adapter (D-049)
        try:
            import torch

            on_gpu = bool(torch.cuda.is_available())
        except ImportError:
            on_gpu = False
        n_m = 20000 if "LaBSE" in m and n_adapter > 20000 and not on_gpu else n_adapter
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


# ------------------------------------------------------------------ v3 sweep (D-050, D-051)

# Order matters (D-051): the pre-registered primary runs first, so a session that dies early
# still produces the gate result.
SWEEP_SPACES = ("massive_full", "massive800", "none")
# (kind, wiki/FLORES pairs, hidden width, + MASSIVE-train English command pairs)
SWEEP_ADAPTERS = (
    ("mlp", 120000, 1024, True),  # primary adapter (D-051)
    ("mlp", 120000, 1024, False),  # D-050's original primary: no command pairs
    ("ridge", 40000, 0, False),  # the v1 recipe
    ("ridge", 120000, 0, True),
    ("mlp", 40000, 1024, True),
    ("mlp", 80000, 1024, True),
    ("mlp", 120000, 2048, True),
)
PRIMARY = {"space": "massive_full", "adapter": "mlp-120000-h1024+cmd", "K1": 256, "D": 256}
# second panel: larger / different models, never part of the gate criterion (D-051)
EXTRA_PANEL = [
    "intfloat/multilingual-e5-base",
    "BAAI/bge-base-en-v1.5",
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
    "mixedbread-ai/mxbai-embed-large-v1",
]
SWEEP_DIMS = (128, 384)  # exploratory meaning-space widths (frozen s1 uses 256)
GRIDS = {
    # the planned run 3: everything above
    "full": {"spaces": SWEEP_SPACES, "adapters": SWEEP_ADAPTERS, "dims": SWEEP_DIMS, "extra": True},
    # a second anchor in the same session (exploratory): reuses every panel embedding
    "anchor2": {
        "spaces": ("massive_full", "none"),
        "adapters": (
            ("mlp", 120000, 1024, True),
            ("mlp", 120000, 1024, False),
            ("ridge", 40000, 0, False),
        ),
        "dims": (),
        "extra": False,
    },
    # GPU smoke test at tiny scale: every code path, a few minutes
    "smoke": {
        "spaces": ("massive_full",),
        "adapters": (("mlp", 120000, 1024, True), ("ridge", 40000, 0, False)),
        "dims": (128,),
        "extra": True,
    },
}


def _bootstrap_ci(agree: np.ndarray, n_boot: int = 1000, seed: int = 0) -> list[float]:
    """95% CI of the mean of a (models, sentences) 0/1 matrix, resampling sentences."""
    rng = np.random.default_rng(seed)
    n = agree.shape[1]
    per_sent = agree.mean(0)
    boots = [per_sent[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    return [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]


def _provenance(models: list[str]) -> dict:
    import platform
    import subprocess

    from sphota.encoders import pinned_revision

    out: dict = {"python": platform.python_version(), "numpy": np.__version__}
    try:
        out["git"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT
        ).stdout.strip()
    except OSError:
        pass
    try:
        import sentence_transformers
        import torch

        out["torch"] = torch.__version__
        out["sentence_transformers"] = sentence_transformers.__version__
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    out["revisions"] = {m: pinned_revision(m, "torch") for m in models}
    return out


def _deadline_passed() -> bool:
    """``SPHOTA_DEADLINE`` (unix seconds): skip remaining blocks so the session can still save."""
    import os

    d = os.environ.get("SPHOTA_DEADLINE")
    return bool(d) and time.time() > float(d)


def v3sweep(
    scale: str,
    anchor: str = "BAAI/bge-m3",
    adapter_models: list[str] | None = None,
    sizes: tuple[int, ...] = (256, 512, 1024),
    seeds: tuple[int, ...] = (0, 1),
    n_max: int = 120000,
    grid: str = "full",
    extra_models: list[str] | None = None,
    out: Path | None = None,
) -> dict:
    """One planned GPU run (D-050, amended by D-051 before any run): embed once, then evaluate
    the gate for every combination of meaning-space scatter x adapter variant x cell count x
    replicate. Replicate r uses k-means seed r AND adapter-training seed r, so the two replicates
    are independent in both random parts. Only the pre-registered PRIMARY combination (grid
    "full", the bge-m3 anchor) is a gate test; both replicates must pass all four criteria.
    Everything else is exploratory. Results are written after every block."""
    from sphota.build.adapters import training_texts
    from sphota.build.adapters_v11 import fit_mlp2
    from sphota.build.common import SCALES
    from sphota.build.release import SPLIT_MEANING, embeddings, load, tatoeba_split
    from sphota.certify import fit_adapter

    G = GRIDS[grid]
    t_start = time.time()
    tag = "" if grid == "full" else f"-{grid}"
    out = out or ROOT / "bench" / "results" / f"v3-sweep-{scale}-{embed.short(anchor)}{tag}.json"
    langs, dev, test, wiki, tat = load(scale)
    E = embeddings(scale, model=anchor)
    keep = diversify(wiki, E["wiki"], max_per_lang=SCALES[scale].get("max_per_lang", 8000))
    fit_ids = np.arange(SPLIT_MEANING)
    cal_ids = np.arange(SPLIT_MEANING, len(dev))
    tr, ca = tatoeba_split(len(tat))
    base_groups = [np.stack([E[("dev", lg)][i] for lg in langs]) for i in fit_ids]
    base_groups += [np.stack([E["tat_x"][i], E["tat_en"][i]]) for i in tr]
    corpus = np.concatenate([E["wiki"][keep]] + [E[("dev", lg)][fit_ids] for lg in langs])
    corpus_langs = np.concatenate(
        [wiki["lang"].values[keep]] + [[lg] * len(fit_ids) for lg in langs]
    )
    mp = massive_parallel(scale, langs, anchor)
    massive_groups = [np.stack([mp["E"][lg][i] for lg in langs]) for i in range(len(mp["ids"]))]
    sub800 = np.sort(
        np.random.default_rng(0).choice(
            len(massive_groups), min(800, len(massive_groups)), replace=False
        )
    )
    # cross-model agreement on short commands (D-051): MASSIVE test, English (held out; the
    # anchor's vectors come from the same cache massive_xling uses)
    mt = massive_parallel(scale, langs, anchor, split="test")
    mt_en = massive_texts(langs, "test", scale)["en"]
    # command pairs for adapter training (D-051): MASSIVE train, English, test texts excluded
    cmd_en = massive_texts(langs, "train", scale)["en"]
    # adapter training pairs: one sample of n_max texts, embedded once per model; smaller training
    # sizes are prefixes of one fixed permutation of it (so 40k is a subset of 80k of 120k)
    texts = training_texts(scale, n_max)
    perm = np.random.default_rng(0).permutation(len(texts))
    X_anchor_tr = embed.cached(anchor, f"adapter-train-{len(texts)}", texts, scale)
    panel = [m for m in (adapter_models or ADAPTER_MODELS) if m != anchor]
    extra = [m for m in (extra_models or EXTRA_PANEL) if m != anchor] if G["extra"] else []
    X_tr, X_te, X_mt, X_cmd = {}, {}, {}, {}
    for m in panel + extra:
        X_cmd[m] = embed.cached(m, f"massive-train-en-{len(cmd_en)}", cmd_en, scale)
        X_tr[m] = embed.cached(m, f"adapter-train-{len(texts)}", texts, scale)
        X_te[m] = embed.cached(m, "flores_devtest_en", test["en"].tolist(), scale)
        X_mt[m] = embed.cached(m, f"massive-test-en-{len(mt_en)}", mt_en, scale)
    log(
        f"v3 sweep [{grid}]: embeddings ready ({time.time() - t_start:.0f}s); panel {len(panel)}"
        f" (+{len(extra)} extra) models, {len(texts)} adapter pairs, "
        f"{len(massive_groups)} MASSIVE meanings"
    )
    is_gate = grid == "full" and anchor == "BAAI/bge-m3" and 256 in sizes
    res: dict = {
        "design": "D-050 + D-051",
        "grid": grid,
        "gate_test": is_gate,
        "anchor": anchor,
        "scale": scale,
        "primary": PRIMARY if is_gate else None,
        "sizes": list(sizes),
        "seeds": list(seeds),
        "panel": panel,
        "extra_panel": extra,
        "n_adapter_pairs": len(texts),
        "n_command_pairs": len(cmd_en),
        "massive_meanings": len(massive_groups),
        "provenance": _provenance([anchor, *panel, *extra]),
        "rows": [],
        "skipped": [],
    }

    def aname(kind, n, h, cmd):
        return f"{kind}-{min(n, len(texts))}" + (f"-h{h}" if h else "") + ("+cmd" if cmd else "")

    def fit(kind, m, idx, h, cmd, seed, Ytr, Ycmd):
        X, Y = X_tr[m][idx], Ytr[idx]
        if cmd:
            X, Y = np.concatenate([X, X_cmd[m]]), np.concatenate([Y, Ycmd])
        if kind == "mlp":
            return fit_mlp2(m, X, Y, hidden=h, seed=seed)
        return fit_adapter(m, X, Y)

    def evaluate(space_name, D, sp, adapters_spec, ks, models_extra):
        Yc = sp(corpus)
        Ya = np.concatenate(
            [sp(E[("dev", "en")][cal_ids])] * (len(langs) - 1) + [sp(E["tat_en"][ca])]
        )
        Yb = np.concatenate(
            [sp(E[("dev", lg)][cal_ids]) for lg in langs[1:]] + [sp(E["tat_x"][ca])]
        )
        Ytest = {lg: sp(E[("test", lg)]) for lg in langs}
        Ytr = sp(X_anchor_tr)
        Ymt = sp(mt["E"]["en"])
        Ycmd = sp(mp["E"]["en"])
        cells = {}
        for k in ks:
            for seed in seeds:
                C = kmeans(Yc, k, seed=seed, iters=40)
                T = calibrate_T(C, Ya, Yb)
                cells[(k, seed)] = (
                    C,
                    T,
                    massive_xling(sp, C, scale, langs, anchor),
                    assign(Yc, C, T)[0],
                )
        log(f"space {space_name} D={D}: fitted, k-means done ({time.time() - t_start:.0f}s)")
        for spec in adapters_spec:
            kind, n, h, cmd = spec
            if _deadline_passed():
                res["skipped"].append(f"{space_name}/D{D}/{aname(*spec)}")
                log(f"deadline: skipped {res['skipped'][-1]}")
                continue
            idx = np.sort(perm[:n])
            ridge_cache: dict = {}
            for seed in seeds:
                ad = {}
                for m in panel + (models_extra if spec == SWEEP_ADAPTERS[0] else []):
                    if kind == "ridge":  # deterministic: fit once, reuse for every replicate
                        if m not in ridge_cache:
                            ridge_cache[m] = fit(kind, m, idx, h, cmd, 0, Ytr, Ycmd)
                        ad[m] = ridge_cache[m]
                    else:
                        ad[m] = fit(kind, m, idx, h, cmd, seed, Ytr, Ycmd)
                for k in ks:
                    C, T, xl, trc = cells[(k, seed)]
                    Yad = {m: ad[m](X_te[m]) for m in panel}
                    met = gate_metrics(C, T, Ytest, Yc, corpus_langs, Yad, tr_cells=trc)
                    v = verdict(met)
                    en_i = assign(Ytest["en"], C, T)[0]
                    agree = np.stack([assign(Yad[m], C, T)[0] == en_i for m in panel])
                    mt_i = assign(Ymt, C, T)[0]
                    mt_agree = {
                        m: float((assign(ad[m](X_mt[m]), C, T)[0] == mt_i).mean()) for m in ad
                    }
                    row = {
                        "space": space_name,
                        "D": D,
                        "adapter": aname(*spec),
                        "K1": k,
                        "seed": seed,
                        "coverage": met["coverage"],
                        "translation_confident": met["translation_confident"],
                        "cross_model": met["cross_model"],
                        "cross_model_ci95": _bootstrap_ci(agree),
                        "cross_model_per_model": met["cross_model_per_model"],
                        "cross_model_commands": round(
                            float(np.mean([mt_agree[m] for m in panel])), 4
                        ),
                        "collapse_max": met["collapse_max"],
                        "collapse_worst_lang": met["collapse_worst_lang"],
                        "massive_same_cell": xl["mean"],
                        "verdict": {k2: {"value": x[0], "pass": bool(x[1])} for k2, x in v.items()},
                    }
                    xm_extra = [m for m in ad if m not in panel]
                    if xm_extra:
                        per = {
                            m.split("/")[-1]: round(
                                float((assign(ad[m](X_te[m]), C, T)[0] == en_i).mean()), 4
                            )
                            for m in xm_extra
                        }
                        row["cross_model_extra"] = round(float(np.mean(list(per.values()))), 4)
                        row["cross_model_extra_per_model"] = per
                        row["cross_model_extra_commands"] = round(
                            float(np.mean([mt_agree[m] for m in xm_extra])), 4
                        )
                    row["passes"] = sum(x["pass"] for x in row["verdict"].values())
                    res["rows"].append(row)
            log(f"  {space_name} D={D} / {aname(*spec)}: done ({time.time() - t_start:.0f}s)")
            out.write_text(json.dumps(res, indent=1), encoding="utf-8")  # partial results

    def space_groups(name):
        groups = list(base_groups)
        if name == "massive800":
            groups += [massive_groups[i] for i in sub800]
        elif name == "massive_full":
            groups += massive_groups
        return groups

    for space_name in G["spaces"]:
        if _deadline_passed():
            res["skipped"].append(space_name)
            continue
        sp = fit_space(corpus, space_groups(space_name), D=256, P=384, reg=0.05)
        evaluate(
            space_name, 256, sp, G["adapters"], sizes, extra if space_name == "massive_full" else []
        )
    # exploratory widths: primary space + primary adapter, 256 cells only
    for D in G["dims"]:
        if _deadline_passed():
            res["skipped"].append(f"D{D}")
            continue
        sp = fit_space(corpus, space_groups("massive_full"), D=D, P=max(384, D * 3 // 2), reg=0.05)
        evaluate("massive_full", D, sp, G["adapters"][:1], tuple(k for k in sizes[:1]), [])
    summarize_sweep(res)
    res["seconds"] = round(time.time() - t_start)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def summarize_sweep(res: dict) -> None:
    """Primary verdict (gate test only) and exploratory leads, in place."""
    seeds = res["seeds"]
    if res.get("gate_test"):
        P = res["primary"]
        prim = [
            r
            for r in res["rows"]
            if r["space"] == P["space"]
            and r["adapter"] == P["adapter"]
            and r["K1"] == P["K1"]
            and r["D"] == P["D"]
        ]
        res["primary_rows"] = prim
        res["primary_gate"] = (
            "PASS (confirmed by both replicates)"
            if len(prim) == len(seeds) and all(r["passes"] == 4 for r in prim)
            else "PASS on one replicate only (not confirmed)"
            if any(r["passes"] == 4 for r in prim)
            else "FAIL"
            if prim
            else "NOT RUN"
        )
    else:
        res["primary_rows"], res["primary_gate"] = [], "not a gate test (exploratory grid)"
    combos: dict[tuple, list[int]] = collections.defaultdict(list)
    for r in res["rows"]:
        combos[(r["space"], r["D"], r["adapter"], r["K1"])].append(r["passes"])
    prim_key = (
        tuple(res["primary"][x] for x in ("space", "D", "adapter", "K1"))
        if res.get("primary")
        else None
    )
    res["exploratory_confirmed"] = [  # all four criteria on every replicate; not the primary
        {"space": c[0], "D": c[1], "adapter": c[2], "K1": c[3]}
        for c, p in combos.items()
        if len(p) == len(seeds) and all(x == 4 for x in p) and c != prim_key and c[3] >= 256
    ]


def print_sweep(res: dict) -> None:
    pct = lambda x: f"{100 * x:5.1f}%"
    print(
        f"\nv3 sweep ({res['anchor']}, {res['scale']}, grid {res['grid']}): "
        f"{len(res['rows'])} gate evaluations, {res.get('seconds', 0) // 60} min"
    )
    print(
        "space         D    adapter            K1    rep  coverage translat cross-mdl [95% CI]"
        "       cmd-xm  collapse MASSIVE  pass"
    )
    for r in res["rows"]:
        star = " <- PRIMARY" if r in res.get("primary_rows", []) else ""
        lo, hi = r["cross_model_ci95"]
        print(
            f"{r['space']:<13} {r['D']:<4} {r['adapter']:<18} {r['K1']:<5} {r['seed']:<4} "
            f"{pct(r['coverage'])}   {pct(r['translation_confident'])}   "
            f"{pct(r['cross_model'])} [{pct(lo)}-{pct(hi)}]  {pct(r['cross_model_commands'])}  "
            f"{pct(r['collapse_max'])}  {pct(r['massive_same_cell'])}  {r['passes']}/4{star}"
        )
    ex = [r for r in res["rows"] if "cross_model_extra" in r]
    for r in ex:
        print(
            f"second panel ({r['space']}, {r['adapter']}, K1={r['K1']}, rep {r['seed']}): "
            f"cross-model {pct(r['cross_model_extra'])} {r['cross_model_extra_per_model']}"
        )
    print(f"\nPRIMARY (D-050/D-051): {res['primary_gate']}")
    if res.get("exploratory_confirmed"):
        print(
            "exploratory combinations passing all four on every replicate at >= 256 cells "
            f"(leads for a new pre-registered run, not gate results): {res['exploratory_confirmed']}"
        )
    if res.get("skipped"):
        print(f"skipped for time: {res['skipped']}")


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
    ap.add_argument("cmd", choices=["sweep", "anchors", "v2mini", "v3sweep"])
    ap.add_argument("--grid", choices=list(GRIDS), default="full", help="v3sweep grid (D-051)")
    ap.add_argument("--extra-models", nargs="*", default=None, help="v3sweep second panel")
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
    elif a.cmd == "v3sweep":
        res = v3sweep(
            a.scale,
            anchor=a.anchor,
            adapter_models=a.adapter_models,
            sizes=tuple(a.sizes),
            n_max=a.n_adapter,
            grid=a.grid,
            extra_models=a.extra_models,
        )
        print_sweep(res)
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
            + (f"-a{a.n_adapter}" if a.n_adapter != 40000 else "")
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
