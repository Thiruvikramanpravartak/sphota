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


def gate_metrics(C, T, Ytest: dict, Ytrain: np.ndarray, train_langs: np.ndarray,
                 Yadapted: dict | None = None) -> dict:
    """Ytest: lang -> (n, D) held-out canonical vectors (row i = same meaning across languages)."""
    langs = list(Ytest)
    A = {lg: assign(Ytest[lg], C, T) for lg in langs}
    conf_all = np.concatenate([A[lg][1] for lg in langs])
    coverage = float((conf_all >= 0.7).mean())
    en_i, en_p, en_c = A["en"]
    m = en_p >= 0.7
    agree = np.mean([(en_i == A[lg][0]).mean() for lg in langs if lg != "en"])
    agree_conf = np.mean([(en_i[m] == A[lg][0][m]).mean() for lg in langs if lg != "en"]) if m.any() else float("nan")
    top2 = np.mean([((A[lg][2][:, :2] == en_i[:, None]).any(1)).mean() for lg in langs if lg != "en"])
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
    collapse = {lg: float(np.mean([dom.get(int(c)) == lg for c in A[lg][0]])) for lg in langs if lg != "en"}
    out = {
        "K1": int(C.shape[0]), "T": T,
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
        "2_translation_confident": (m["translation_confident"], m["translation_confident"] >= GATE["translation_confident"]),
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
        log(f"K1={k}: cov {m['coverage']} tr {m['translation_all']} tr|conf {m['translation_confident']} "
            f"xm {m['cross_model']} collapse {m['collapse_max']} top2 {m['translation_top2']}")
        res.append(m)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sweep"])
    ap.add_argument("--scale", default="small")
    a = ap.parse_args()
    if a.cmd == "sweep":
        res = sweep(a.scale)
        (ROOT / "bench" / "results" / "l1-sweep.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
