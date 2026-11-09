# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-file refinement study, run exactly as pre-registered in D-033.

    python -m sphota.build.refine_study --scale small

1. Every labelled dataset × language is split once into dev (30%) and test (70%), seed 20261109.
2. On **dev only**: candidates M1 (k-means, silhouette-chosen k), M2 (bisect, threshold ``tau``
   in translation units) and M3 (split-and-merge of fixed themes, threshold ``tau``) are swept;
   the winner (method + ``tau``) maximizes the mean of (V-measure − reference) over all dev sets.
3. On **test, once**: the winner and the other candidates at their best dev setting are evaluated
   against targets R1–R4, with 95% bootstrap CIs (1,000 resamples).

Writes ``bench/results/refine-study-<scale>.json``. Nothing here looks at test labels before the
method is fixed; the script enforces the order (dev selection runs first and is frozen).
"""

from __future__ import annotations

import argparse
import gzip
import json
import time

import numpy as np

from sphota.build import embed
from sphota.build.common import ROOT, log
from sphota.build.domains import (
    ANCHOR,
    _first_split_jsonl,
    _hf,
    load_banking,
    load_indic,
    load_massive,
    nearest,
)

SEED = 20261109
MASSIVE_EVAL = ["en", "hi", "ta", "te", "bn", "es", "zh-CN", "ar"]
INDIC_EVAL = ["hi", "ta", "bn", "te", "mr"]
MASAKHA_EVAL = ["eng", "fra", "swa", "hau", "amh", "yor"]
TAUS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4]
KS = [4, 8, 16, 32, 64, 128, 256, 512]
B = 1000


def split(n: int) -> tuple[np.ndarray, np.ndarray]:
    perm = np.random.default_rng(SEED).permutation(n)
    k = round(0.3 * n)
    return np.sort(perm[:k]), np.sort(perm[k:])


def load_masakha(lg: str) -> tuple[list[str], list[str]]:
    import pandas as pd

    df = pd.read_parquet(_hf("mteb/MasakhaNEWSClusteringS2S", f"{lg}/test-00000-of-00001.parquet"))
    S = [s for i in range(len(df)) for s in df.sentences[i]]
    L = [str(x) for i in range(len(df)) for x in df.labels[i]]
    seen, keep = set(), []
    for i, s in enumerate(S):  # the five MTEB splits overlap; keep each headline once
        if s not in seen:
            seen.add(s)
            keep.append(i)
    return [S[i] for i in keep], [L[i] for i in keep]


def datasets(scale: str) -> dict[str, tuple[list[str], list[str]]]:
    """name -> (texts, labels). Same loaders and sizes as build.domains (small scale)."""
    out = {}
    n_mas, n_flat, n_ind = (300, 600, 200) if scale == "tiny" else (2974, 5000, 1000)
    mas = load_massive(n_mas)
    for lg in MASSIVE_EVAL:
        out[f"massive/{lg}"] = mas[lg]
    out["banking77/en"] = load_banking(n_flat)
    out["stackexchange/en"] = _first_split_jsonl(
        "mteb/stackexchange-clustering", "test.jsonl", n_flat
    )
    out["arxiv/en"] = _first_split_jsonl("mteb/arxiv-clustering-s2s", "test.jsonl.gz", n_flat)
    ind = load_indic(n_ind)
    for lg in INDIC_EVAL:
        out[f"indic_reviews/{lg}"] = ind[lg]
    for lg in MASAKHA_EVAL:
        t, lab = load_masakha(lg)
        if scale == "tiny":
            t, lab = t[:300], lab[:300]
        out[f"masakhanews/{lg}"] = (t, lab)
    return out


# ------------------------------------------------------------------ methods

# a candidate: (method, tau, language); method in M1_kmeans_silhouette | M2_bisect |
# M3_split_merge | M3_auto (split-and-merge, tau chosen per file by silhouette)
Candidate = tuple[str, float | None, str]


def key_of(c: Candidate) -> str:
    m, t, lang = c
    return m + (f"@tau={t}" if t is not None else "") + (f"+{lang}" if lang != "none" else "")


def run_method(c: Candidate, Y: np.ndarray, l1: np.ndarray, scripts: np.ndarray) -> np.ndarray:
    from sphota.refine import refine

    m, t, lang = c
    if m.startswith("M4_kmeans_"):  # round 3: M4_kmeans_<k_rule>
        from sphota.refine import refine as _r

        return _r(Y, "kmeans", scripts=scripts, language=lang, k_rule=m.split("_", 2)[2]).labels
    if m == "M1_kmeans_silhouette":
        return kmeans_silhouette(Y)
    method = {
        "M2_bisect": "bisect",
        "M3_split_merge": "split_merge",
        "M3_auto": "auto",
        "M4_kmeans": "kmeans",
    }[m]
    return refine(Y, method, tau=t, l1=l1, scripts=scripts, language=lang).labels


def kmeans_silhouette(Y: np.ndarray) -> np.ndarray:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    sub = np.random.default_rng(SEED).choice(len(Y), min(2000, len(Y)), replace=False)
    best, best_s = None, -2.0
    for k in [k for k in KS if k < len(Y) / 3]:
        lab = KMeans(k, n_init=2, random_state=0).fit_predict(Y)
        s = silhouette_score(Y[sub], lab[sub])
        if s > best_s:
            best, best_s = lab, s
    assert best is not None
    return best


def reference(X: np.ndarray, labels: list[str], k_pred: int) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import v_measure_score

    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    out = {}
    for name, k in [("kmeans_k_true", len(set(labels))), ("kmeans_same_groups", k_pred)]:
        k = max(2, min(k, len(X) - 1))
        pred = KMeans(k, n_init=3, random_state=0).fit_predict(Xn)
        out[name] = float(v_measure_score(labels, pred))
    out["reference"] = max(out.values())
    return out


def vm(labels, pred) -> float:
    from sklearn.metrics import v_measure_score

    return float(v_measure_score(labels, pred))


def boot_ci(labels: list[str], pred: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    lab = np.asarray(labels)
    vals = [vm(lab[i], pred[i]) for i in (rng.integers(0, len(lab), len(lab)) for _ in range(B))]
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


R3_PAIRS = [("massive", "en", lg) for lg in ["hi", "ta", "es", "zh-CN", "ar"]] + [
    ("indic_reviews", "hi", lg) for lg in ["ta", "bn", "te"]
]


def r3_eval(c: Candidate, part: int, data, emb, splits) -> dict:
    """Mixed-language files from parallel items of one split (0 = dev, 1 = test)."""
    out = {}
    for ds, a, b in R3_PAIRS:
        na, nb = f"{ds}/{a}", f"{ds}/{b}"
        assert len(data[na][0]) == len(data[nb][0])  # parallel rows, same split
        ix = splits[na][part]
        Ya, Yb = emb[na][1][ix], emb[nb][1][ix]
        la, lb = emb[na][2][ix], emb[nb][2][ix]
        sa, sb = emb[na][3][ix], emb[nb][3][ix]
        pred = run_method(
            c, np.concatenate([Ya, Yb]), np.concatenate([la, lb]), np.concatenate([sa, sb])
        )
        k = len(ix)
        same = float((pred[:k] == pred[k:]).mean())
        theme = float((la == lb).mean())
        out[f"{na}+{b}"] = {
            "same_refined_group": round(same, 4),
            "same_fixed_theme": round(theme, 4),
            "pass": bool(same >= theme),
        }
    return out


# ------------------------------------------------------------------ study


def run(scale: str) -> dict:
    from sklearn.metrics import adjusted_rand_score

    from sphota.refine import script_of
    from sphota.release import load_release
    from sphota.space import project

    rel = load_release()
    C1 = rel.codebooks[0].astype(np.float64)
    data = datasets(scale)
    emb = {}
    for name, (texts, _labels) in data.items():
        X = embed.cached(ANCHOR, f"refine-{name.replace('/', '-')}-{len(texts)}", texts, scale)
        Y = project(rel, X).astype(np.float64)
        emb[name] = (X, Y, nearest(Y, C1), np.array([script_of(t) for t in texts]))
    splits = {name: split(len(data[name][0])) for name in data}

    # ---- 1. dev selection: R1 gaps per (method, tau); R3 per (method, tau, language)
    t0 = time.time()
    base = [("M1_kmeans_silhouette", None, "none"), ("M3_auto", None, "none")] + [
        (m, t, "none") for m in ["M2_bisect", "M3_split_merge"] for t in TAUS
    ]
    dev: dict = {}
    for c in base:
        gaps, rows = [], {}
        for name, (_texts, labels) in data.items():
            d = splits[name][0]
            X, Y, l1, sc = (a[d] for a in emb[name])
            lab = [labels[i] for i in d]
            pred = run_method(c, Y, l1, sc)
            ref = reference(X, lab, len(set(pred.tolist())))
            v = vm(lab, pred)
            gaps.append(v - ref["reference"])
            rows[name] = {
                "v": round(v, 4),
                "ref": round(ref["reference"], 4),
                "groups": len(set(pred.tolist())),
            }
        for lang in ["none", "centre", "guard"] if c[0] != "M1_kmeans_silhouette" else ["none"]:
            cl = (c[0], c[1], lang)
            r3 = r3_eval(cl, 0, data, emb, splits)
            dev[key_of(cl)] = {
                "candidate": list(cl),
                "mean_gap": round(float(np.mean(gaps)), 4),
                "min_gap": round(float(np.min(gaps)), 4),
                "per_dataset": rows,
                "R3_dev": r3,
                "R3_dev_pass": all(x["pass"] for x in r3.values()),
            }
            log(
                f"dev {key_of(cl):<34} gap {np.mean(gaps):+.4f} worst {np.min(gaps):+.4f}  "
                f"R3 {sum(x['pass'] for x in r3.values())}/{len(r3)}"
            )
    # selection rule (D-033 + amendment D-034): eligible = passes R3 on dev; maximize mean gap
    eligible = {k: v for k, v in dev.items() if v["R3_dev_pass"]} or dev
    best_per_method: dict[str, str] = {}
    for k, v in eligible.items():
        m = v["candidate"][0]
        if m not in best_per_method or v["mean_gap"] > eligible[best_per_method[m]]["mean_gap"]:
            best_per_method[m] = k
    winner = max(best_per_method.values(), key=lambda k: eligible[k]["mean_gap"])
    log(
        f"dev selection frozen: winner {winner}; per method {best_per_method} ({time.time() - t0:.0f}s)"
    )

    # ---- 2. test, once per method at its best eligible dev setting
    test: dict = {}
    for k in best_per_method.values():
        c = tuple(dev[k]["candidate"])
        per = {}
        for name, (_texts, labels) in data.items():
            te = splits[name][1]
            X, Y, l1, sc = (a[te] for a in emb[name])
            lab = [labels[i] for i in te]
            t1 = time.time()
            pred = run_method(c, Y, l1, sc)
            secs = time.time() - t1
            ref = reference(X, lab, len(set(pred.tolist())))
            v = vm(lab, pred)
            perm = np.random.default_rng(1).permutation(len(Y))
            ari = adjusted_rand_score(pred[perm], run_method(c, Y[perm], l1[perm], sc[perm]))
            per[name] = {
                "n": len(te),
                "labels": len(set(lab)),
                "groups": len(set(pred.tolist())),
                "refined_v": round(v, 4),
                "refined_v_ci": boot_ci(lab, pred),
                "themes_v": round(vm(lab, l1), 4),
                "themes_v_ci": boot_ci(lab, l1),
                **{kk: round(x, 4) for kk, x in ref.items()},
                "R1_pass": bool(v >= ref["reference"] - 0.05),
                "shuffle_ari": round(float(ari), 4),
                "seconds": round(secs, 3),
            }
        r3 = r3_eval(c, 1, data, emb, splits)
        test[k] = {
            "per_dataset": per,
            "R1_all": all(x["R1_pass"] for x in per.values()),
            "R1_passed": sum(x["R1_pass"] for x in per.values()),
            "datasets": len(per),
            "R2_shuffle_min_ari": min(x["shuffle_ari"] for x in per.values()),
            "R2_pass": min(x["shuffle_ari"] for x in per.values()) >= 0.95,
            "R3": r3,
            "R3_pass": all(x["pass"] for x in r3.values()),
        }
        log(
            f"test {k}: R1 {test[k]['R1_passed']}/{len(per)}  R2 min ARI {test[k]['R2_shuffle_min_ari']}  "
            f"R3 {sum(x['pass'] for x in r3.values())}/{len(r3)}"
        )

    # ---- 3. R4 speed (winner)
    c = tuple(dev[winner]["candidate"])
    names = ["massive/en", "banking77/en"]
    big = np.concatenate([emb[n][1] for n in names])[:3000]
    l1big = np.concatenate([emb[n][2] for n in names])[:3000]
    scbig = np.concatenate([emb[n][3] for n in names])[:3000]
    times = []
    for _ in range(3):
        t1 = time.time()
        run_method(c, big, l1big, scbig)
        times.append(time.time() - t1)
    r4 = {
        "rows": len(big),
        "python_seconds_median": round(float(np.median(times)), 3),
        "R4_python_pass": bool(np.median(times) <= 2.0),
    }
    return {
        "scale": scale,
        "seed": SEED,
        "pre_registration": "DECISIONS.md D-033, amendment D-034",
        "dev": dev,
        "best_per_method": best_per_method,
        "winner": winner,
        "test": test,
        "R2_codes_unchanged": "by construction: refinement never alters a message's code",
        "R4": r4,
    }


# ------------------------------------------------------------------ round 2 (D-036)

CONFIRM_MASSIVE = ["de", "fr", "ru", "ur", "kn", "ml"]
CONFIRM_INDIC = ["gu", "kn", "ml", "or", "pa", "ur"]
CONFIRM_MASAKHA = ["ibo", "lin", "lug", "orm", "pcm", "run", "sna", "som", "tir", "xho"]
CONFIRM_R3 = [("massive", "en", lg) for lg in ["de", "fr", "ru", "ur"]] + [
    ("indic_reviews", "hi", lg) for lg in ["gu", "kn", "ml"]
]


def _massive_langs(langs: list[str]) -> dict[str, tuple[list[str], list[str]]]:
    out = {}
    for lg in langs:
        with gzip.open(_hf("mteb/amazon_massive_intent", f"test/{lg}.json.gz"), "rt") as f:
            rows = sorted((json.loads(line) for line in f), key=lambda r: int(r["id"]))
        out[lg] = ([r["text"] for r in rows], [r["label"] for r in rows])
    return out


def _jsonl_split(repo: str, fname: str, line: int, n: int) -> tuple[list[str], list[str]]:
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(_hf(repo, fname), "rt", encoding="utf-8") as f:
        for i, row in enumerate(f):
            if i == line:
                j = json.loads(row)
                break
    idx = np.sort(
        np.random.default_rng(SEED).choice(
            len(j["sentences"]), min(n, len(j["sentences"])), replace=False
        )
    )
    return [j["sentences"][i] for i in idx], [str(j["labels"][i]) for i in idx]


def confirmation_sets() -> dict[str, tuple[list[str], list[str]]]:
    import pandas as pd

    from sphota.build.domains import INDIC_LANGS

    out = {}
    for lg, v in _massive_langs(["en", *CONFIRM_MASSIVE]).items():
        out[f"massive/{lg}"] = v
    ind = load_indic(1000)
    for lg in ["hi", *CONFIRM_INDIC]:
        assert lg in INDIC_LANGS
        out[f"indic_reviews/{lg}"] = ind[lg]
    for lg in CONFIRM_MASAKHA:
        out[f"masakhanews/{lg}"] = load_masakha(lg)
    out["stackexchange#2/en"] = _jsonl_split("mteb/stackexchange-clustering", "test.jsonl", 1, 5000)
    out["arxiv#2/en"] = _jsonl_split("mteb/arxiv-clustering-s2s", "test.jsonl.gz", 1, 5000)
    df = pd.read_parquet(_hf("PolyAI/banking77", "data/train-00000-of-00001.parquet"))
    idx = np.sort(np.random.default_rng(SEED).choice(len(df), 3000, replace=False))
    out["banking77-train/en"] = (df.text.iloc[idx].tolist(), [str(x) for x in df.label.iloc[idx]])
    return out


def round2(scale: str) -> dict:
    from sklearn.metrics import adjusted_rand_score

    from sphota.refine import script_of
    from sphota.release import load_release
    from sphota.space import project

    rel = load_release()
    C1 = rel.codebooks[0].astype(np.float64)

    def embed_all(data: dict, prefix: str) -> dict:
        emb = {}
        for name, (texts, _labels) in data.items():
            X = embed.cached(
                ANCHOR,
                f"{prefix}-{name.replace('/', '-').replace('#', '')}-{len(texts)}",
                texts,
                scale,
            )
            Y = project(rel, X).astype(np.float64)
            emb[name] = (X, Y, nearest(Y, C1), np.array([script_of(t) for t in texts]))
        return emb

    # ---- selection on the round-1 dev splits (D-034 rule)
    data = datasets(scale)
    emb = embed_all(data, "refine")
    splits = {name: split(len(data[name][0])) for name in data}
    cands = [("M3_auto", None, lang) for lang in ["none", "centre", "guard"]] + [
        ("M4_kmeans", None, lang) for lang in ["none", "centre"]
    ]
    dev = {}
    for c in cands:
        gaps = []
        for name, (_texts, labels) in data.items():
            d = splits[name][0]
            X, Y, l1, sc = (a[d] for a in emb[name])
            lab = [labels[i] for i in d]
            pred = run_method(c, Y, l1, sc)
            gaps.append(vm(lab, pred) - reference(X, lab, len(set(pred.tolist())))["reference"])
        r3 = r3_eval(c, 0, data, emb, splits)
        dev[key_of(c)] = {
            "candidate": list(c),
            "mean_gap": round(float(np.mean(gaps)), 4),
            "min_gap": round(float(np.min(gaps)), 4),
            "R3_dev_pass": all(x["pass"] for x in r3.values()),
            "R3_dev": {k: v["pass"] for k, v in r3.items()},
        }
        log(
            f"round2 dev {key_of(c):<22} gap {np.mean(gaps):+.4f} worst {np.min(gaps):+.4f} "
            f"R3 {sum(x['pass'] for x in r3.values())}/{len(r3)}"
        )
    eligible = {k: v for k, v in dev.items() if v["R3_dev_pass"]} or dev
    winner = max(eligible, key=lambda k: eligible[k]["mean_gap"])
    c = tuple(dev[winner]["candidate"])
    log(f"round2 selection frozen: {winner}")

    # ---- the winner, once, on the fresh confirmation set
    conf = confirmation_sets()
    cemb = embed_all(conf, "confirm")
    per = {}
    for name, (_texts, labels) in conf.items():
        if name in ("massive/en", "indic_reviews/hi"):
            continue  # pivots for R3 only (already used in round 1)
        X, Y, l1, sc = cemb[name]
        pred = run_method(c, Y, l1, sc)
        ref = reference(X, labels, len(set(pred.tolist())))
        v = vm(labels, pred)
        perm = np.random.default_rng(1).permutation(len(Y))
        ari = adjusted_rand_score(pred[perm], run_method(c, Y[perm], l1[perm], sc[perm]))
        per[name] = {
            "n": len(labels),
            "labels": len(set(labels)),
            "groups": len(set(pred.tolist())),
            "refined_v": round(v, 4),
            "refined_v_ci": boot_ci(labels, pred),
            "themes_v": round(vm(labels, l1), 4),
            **{k: round(x, 4) for k, x in ref.items()},
            "R1_pass": bool(v >= ref["reference"] - 0.05),
            "shuffle_ari": round(float(ari), 4),
        }
        log(
            f"confirm {name:<22} V {v:.4f} ref {ref['reference']:.4f} {'PASS' if per[name]['R1_pass'] else 'miss'}"
        )
    r3 = {}
    for ds, a, b in CONFIRM_R3:
        na, nb = f"{ds}/{a}", f"{ds}/{b}"
        Ya, Yb = cemb[na][1], cemb[nb][1]
        la, lb = cemb[na][2], cemb[nb][2]
        sa, sb = cemb[na][3], cemb[nb][3]
        pred = run_method(
            c, np.concatenate([Ya, Yb]), np.concatenate([la, lb]), np.concatenate([sa, sb])
        )
        k = len(Ya)
        same, theme = float((pred[:k] == pred[k:]).mean()), float((la == lb).mean())
        r3[f"{na}+{b}"] = {
            "same_refined_group": round(same, 4),
            "same_fixed_theme": round(theme, 4),
            "pass": bool(same >= theme),
        }
    big = np.concatenate([cemb[n][1] for n in ["massive/de", "banking77-train/en"]])[:3000]
    l1b = nearest(big, C1)
    scb = np.concatenate([cemb[n][3] for n in ["massive/de", "banking77-train/en"]])[:3000]
    times = []
    for _ in range(3):
        t1 = time.time()
        run_method(c, big, l1b, scb)
        times.append(time.time() - t1)
    res = {
        "round": 2,
        "pre_registration": "DECISIONS.md D-036",
        "dev": dev,
        "winner": winner,
        "confirmation": per,
        "R1_passed": sum(x["R1_pass"] for x in per.values()),
        "R1_sets": len(per),
        "R1_all": all(x["R1_pass"] for x in per.values()),
        "R2_min_shuffle_ari": min(x["shuffle_ari"] for x in per.values()),
        "R3": r3,
        "R3_all": all(x["pass"] for x in r3.values()),
        "R4": {
            "rows": len(big),
            "python_seconds_median": round(float(np.median(times)), 3),
            "R4_python_pass": bool(np.median(times) <= 2.0),
        },
    }
    res["R2_pass"] = res["R2_min_shuffle_ari"] >= 0.95
    return res


# ------------------------------------------------------------------ round 3 (D-037)

CONFIRM2_MASSIVE = ["it", "pt", "nl", "pl", "tr", "vi", "id", "ja", "ko", "fa", "he", "th", "sw"]
CONFIRM2_R3 = [("massive", "en", lg) for lg in ["it", "ja", "tr", "sw"]]


def confirmation_sets_2() -> dict[str, tuple[list[str], list[str]]]:
    import pandas as pd

    out = {f"massive/{lg}": v for lg, v in _massive_langs(["en", *CONFIRM2_MASSIVE]).items()}
    out["stackexchange#3/en"] = _jsonl_split("mteb/stackexchange-clustering", "test.jsonl", 2, 5000)
    out["arxiv#3/en"] = _jsonl_split("mteb/arxiv-clustering-s2s", "test.jsonl.gz", 2, 5000)
    df = pd.read_parquet(_hf("PolyAI/banking77", "data/train-00000-of-00001.parquet"))
    used = set(np.random.default_rng(SEED).choice(len(df), 3000, replace=False).tolist())
    rest = np.array([i for i in range(len(df)) if i not in used])
    idx = np.sort(np.random.default_rng(SEED + 1).choice(rest, 3000, replace=False))
    out["banking77-train-b/en"] = (df.text.iloc[idx].tolist(), [str(x) for x in df.label.iloc[idx]])
    return out


def round3(scale: str) -> dict:
    from sklearn.metrics import adjusted_rand_score

    from sphota.refine import script_of
    from sphota.release import load_release
    from sphota.space import project

    rel = load_release()
    C1 = rel.codebooks[0].astype(np.float64)

    def embed_all(data: dict, prefix: str) -> dict:
        emb = {}
        for name, (texts, _labels) in data.items():
            tag = name.replace("/", "-").replace("#", "")
            X = embed.cached(ANCHOR, f"{prefix}-{tag}-{len(texts)}", texts, scale)
            Y = project(rel, X).astype(np.float64)
            emb[name] = (X, Y, nearest(Y, C1), np.array([script_of(t) for t in texts]))
        return emb

    data = datasets(scale)
    emb = embed_all(data, "refine")
    splits = {name: split(len(data[name][0])) for name in data}
    cands = [(f"M4_kmeans_{r}", None, "centre") for r in ["silhouette", "sqrt", "2sqrt", "max"]]
    dev = {}
    for c in cands:
        gaps = []
        for name, (_texts, labels) in data.items():
            d = splits[name][0]
            X, Y, l1, sc = (a[d] for a in emb[name])
            lab = [labels[i] for i in d]
            pred = run_method(c, Y, l1, sc)
            gaps.append(vm(lab, pred) - reference(X, lab, len(set(pred.tolist())))["reference"])
        r3 = r3_eval(c, 0, data, emb, splits)
        dev[key_of(c)] = {
            "candidate": list(c),
            "mean_gap": round(float(np.mean(gaps)), 4),
            "min_gap": round(float(np.min(gaps)), 4),
            "R1_dev_passed": int(sum(g >= -0.05 for g in gaps)),
            "dev_sets": len(gaps),
            "R3_dev_pass": all(x["pass"] for x in r3.values()),
        }
        log(
            f"round3 dev {key_of(c):<28} gap {np.mean(gaps):+.4f} worst {np.min(gaps):+.4f} "
            f"R1dev {dev[key_of(c)]['R1_dev_passed']}/{len(gaps)} R3 {sum(x['pass'] for x in r3.values())}/{len(r3)}"
        )
    eligible = {k: v for k, v in dev.items() if v["R3_dev_pass"]} or dev
    winner = max(eligible, key=lambda k: eligible[k]["mean_gap"])
    c = tuple(dev[winner]["candidate"])
    log(f"round3 selection frozen: {winner}")

    conf = confirmation_sets_2()
    cemb = embed_all(conf, "confirm2")
    per = {}
    for name, (_texts, labels) in conf.items():
        if name == "massive/en":
            continue
        X, Y, l1, sc = cemb[name]
        pred = run_method(c, Y, l1, sc)
        ref = reference(X, labels, len(set(pred.tolist())))
        v = vm(labels, pred)
        perm = np.random.default_rng(1).permutation(len(Y))
        ari = adjusted_rand_score(pred[perm], run_method(c, Y[perm], l1[perm], sc[perm]))
        per[name] = {
            "n": len(labels),
            "labels": len(set(labels)),
            "groups": len(set(pred.tolist())),
            "refined_v": round(v, 4),
            "refined_v_ci": boot_ci(labels, pred),
            "themes_v": round(vm(labels, l1), 4),
            **{k: round(x, 4) for k, x in ref.items()},
            "R1_pass": bool(v >= ref["reference"] - 0.05),
            "shuffle_ari": round(float(ari), 4),
        }
        log(
            f"confirm2 {name:<22} V {v:.4f} ref {ref['reference']:.4f} groups {per[name]['groups']} "
            f"{'PASS' if per[name]['R1_pass'] else 'miss'}"
        )
    r3 = {}
    for ds, a, b in CONFIRM2_R3:
        na, nb = f"{ds}/{a}", f"{ds}/{b}"
        Ya, Yb = cemb[na][1], cemb[nb][1]
        la, lb = cemb[na][2], cemb[nb][2]
        sa, sb = cemb[na][3], cemb[nb][3]
        pred = run_method(
            c, np.concatenate([Ya, Yb]), np.concatenate([la, lb]), np.concatenate([sa, sb])
        )
        k = len(Ya)
        same, theme = float((pred[:k] == pred[k:]).mean()), float((la == lb).mean())
        r3[f"{na}+{b}"] = {
            "same_refined_group": round(same, 4),
            "same_fixed_theme": round(theme, 4),
            "pass": bool(same >= theme),
        }
    big = np.concatenate([cemb[n][1] for n in ["massive/it", "banking77-train-b/en"]])[:3000]
    scb = np.concatenate([cemb[n][3] for n in ["massive/it", "banking77-train-b/en"]])[:3000]
    l1b = nearest(big, C1)
    times = []
    for _ in range(3):
        t1 = time.time()
        run_method(c, big, l1b, scb)
        times.append(time.time() - t1)
    res = {
        "round": 3,
        "pre_registration": "DECISIONS.md D-037",
        "dev": dev,
        "winner": winner,
        "confirmation": per,
        "R1_passed": sum(x["R1_pass"] for x in per.values()),
        "R1_sets": len(per),
        "R1_all": all(x["R1_pass"] for x in per.values()),
        "R2_min_shuffle_ari": min(x["shuffle_ari"] for x in per.values()),
        "R3": r3,
        "R3_all": all(x["pass"] for x in r3.values()),
        "R4": {
            "rows": len(big),
            "python_seconds_median": round(float(np.median(times)), 3),
            "R4_python_pass": bool(np.median(times) <= 2.0),
        },
    }
    res["R2_pass"] = res["R2_min_shuffle_ari"] >= 0.95
    return res


# ------------------------------------------------------------------ round 4 (D-038)

CONFIRM3_MASSIVE = [
    "af",
    "am",
    "az",
    "cy",
    "da",
    "el",
    "fi",
    "hu",
    "hy",
    "is",
    "jv",
    "ka",
    "km",
    "lv",
    "mn",
    "ms",
    "my",
    "nb",
    "ro",
    "sl",
    "sq",
    "sv",
    "tl",
    "zh-TW",
]
CONFIRM3_R3 = [("massive", "en", lg) for lg in ["da", "ro", "sv", "ms", "el", "ka"]]
R4_CANDS = [
    ("canonical", "script"),
    ("canonical", "language"),
    ("raw", "language"),
    ("raw", "script"),
]


def r4_run(c: tuple[str, str], X: np.ndarray, Y: np.ndarray, keys: dict) -> np.ndarray:
    from sphota.refine import refine

    space, key = c
    V = Y if space == "canonical" else X / np.linalg.norm(X, axis=1, keepdims=True)
    return refine(V, "kmeans", scripts=keys[key], language="centre", k_rule="2sqrt").labels


def confirmation_sets_3() -> dict[str, tuple[list[str], list[str]]]:
    import pandas as pd

    out = {f"massive/{lg}": v for lg, v in _massive_langs(["en", *CONFIRM3_MASSIVE]).items()}
    out["stackexchange#4/en"] = _jsonl_split("mteb/stackexchange-clustering", "test.jsonl", 3, 5000)
    out["arxiv#4/en"] = _jsonl_split("mteb/arxiv-clustering-s2s", "test.jsonl.gz", 3, 5000)
    df = pd.read_parquet(_hf("PolyAI/banking77", "data/train-00000-of-00001.parquet"))
    used = set(np.random.default_rng(SEED).choice(len(df), 3000, replace=False).tolist())
    rest = np.array([i for i in range(len(df)) if i not in used])
    used |= set(np.random.default_rng(SEED + 1).choice(rest, 3000, replace=False).tolist())
    rest = np.array([i for i in range(len(df)) if i not in used])
    idx = np.sort(np.random.default_rng(SEED + 2).choice(rest, min(3000, len(rest)), replace=False))
    out["banking77-train-c/en"] = (df.text.iloc[idx].tolist(), [str(x) for x in df.label.iloc[idx]])
    return out


def round4(scale: str) -> dict:
    from sklearn.metrics import adjusted_rand_score

    from sphota.langid import language_keys
    from sphota.refine import script_of
    from sphota.release import load_release
    from sphota.space import project

    rel = load_release()
    C1 = rel.codebooks[0].astype(np.float64)

    def embed_all(data: dict, prefix: str) -> dict:
        emb = {}
        for name, (texts, _labels) in data.items():
            tag = name.replace("/", "-").replace("#", "")
            X = embed.cached(ANCHOR, f"{prefix}-{tag}-{len(texts)}", texts, scale)
            Y = project(rel, X).astype(np.float64)
            keys = {
                "script": np.array([script_of(t) for t in texts]),
                "language": language_keys(texts, X),
            }
            emb[name] = (X, Y, nearest(Y, C1), keys)
        return emb

    def sub(keys: dict, ix) -> dict:
        return {k: v[ix] for k, v in keys.items()}

    def cat(a: dict, b: dict) -> dict:
        return {k: np.concatenate([a[k], b[k]]) for k in a}

    data = datasets(scale)
    emb = embed_all(data, "refine")
    splits = {name: split(len(data[name][0])) for name in data}
    dev = {}
    for c in R4_CANDS:
        gaps = []
        for name, (_texts, labels) in data.items():
            d = splits[name][0]
            X, Y, _l1, keys = emb[name]
            lab = [labels[i] for i in d]
            pred = r4_run(c, X[d], Y[d], sub(keys, d))
            gaps.append(vm(lab, pred) - reference(X[d], lab, len(set(pred.tolist())))["reference"])
        r3p = 0
        for ds, a, b in R3_PAIRS:
            na, nb = f"{ds}/{a}", f"{ds}/{b}"
            ix = splits[na][0]
            Xa, Ya, la, ka = emb[na]
            Xb, Yb, lb, kb = emb[nb]
            pred = r4_run(
                c,
                np.concatenate([Xa[ix], Xb[ix]]),
                np.concatenate([Ya[ix], Yb[ix]]),
                cat(sub(ka, ix), sub(kb, ix)),
            )
            k = len(ix)
            r3p += int((pred[:k] == pred[k:]).mean() >= (la[ix] == lb[ix]).mean())
        key = f"M4_kmeans_2sqrt[{c[0]},{c[1]}]"
        dev[key] = {
            "candidate": list(c),
            "mean_gap": round(float(np.mean(gaps)), 4),
            "min_gap": round(float(np.min(gaps)), 4),
            "R1_dev_passed": int(sum(g >= -0.05 for g in gaps)),
            "dev_sets": len(gaps),
            "R3_dev_passed": r3p,
            "R3_dev_pass": r3p == len(R3_PAIRS),
        }
        log(
            f"round4 dev {key:<36} gap {np.mean(gaps):+.4f} worst {np.min(gaps):+.4f} "
            f"R1dev {dev[key]['R1_dev_passed']}/{len(gaps)} R3 {r3p}/{len(R3_PAIRS)}"
        )
    eligible = {k: v for k, v in dev.items() if v["R3_dev_pass"]} or dev
    winner = max(eligible, key=lambda k: eligible[k]["mean_gap"])
    c = tuple(dev[winner]["candidate"])
    log(f"round4 selection frozen: {winner}")

    conf = confirmation_sets_3()
    cemb = embed_all(conf, "confirm3")
    per = {}
    for name, (_texts, labels) in conf.items():
        if name == "massive/en":
            continue
        X, Y, l1, keys = cemb[name]
        pred = r4_run(c, X, Y, keys)
        ref = reference(X, labels, len(set(pred.tolist())))
        v = vm(labels, pred)
        perm = np.random.default_rng(1).permutation(len(Y))
        ari = adjusted_rand_score(pred[perm], r4_run(c, X[perm], Y[perm], sub(keys, perm)))
        per[name] = {
            "n": len(labels),
            "labels": len(set(labels)),
            "groups": len(set(pred.tolist())),
            "refined_v": round(v, 4),
            "refined_v_ci": boot_ci(labels, pred),
            "themes_v": round(vm(labels, l1), 4),
            **{k: round(x, 4) for k, x in ref.items()},
            "R1_pass": bool(v >= ref["reference"] - 0.05),
            "shuffle_ari": round(float(ari), 4),
        }
        log(
            f"confirm3 {name:<22} V {v:.4f} ref {ref['reference']:.4f} {'PASS' if per[name]['R1_pass'] else 'miss'}"
        )
    r3 = {}
    for ds, a, b in CONFIRM3_R3:
        na, nb = f"{ds}/{a}", f"{ds}/{b}"
        Xa, Ya, la, ka = cemb[na]
        Xb, Yb, lb, kb = cemb[nb]
        pred = r4_run(c, np.concatenate([Xa, Xb]), np.concatenate([Ya, Yb]), cat(ka, kb))
        k = len(Ya)
        same, theme = float((pred[:k] == pred[k:]).mean()), float((la == lb).mean())
        r3[f"{na}+{b}"] = {
            "same_refined_group": round(same, 4),
            "same_fixed_theme": round(theme, 4),
            "pass": bool(same >= theme),
        }
    names = ["massive/da", "banking77-train-c/en"]
    Xbig = np.concatenate([cemb[n][0] for n in names])[:3000]
    Ybig = np.concatenate([cemb[n][1] for n in names])[:3000]
    kbig = {
        k: np.concatenate([cemb[n][3][k] for n in names])[:3000] for k in ("script", "language")
    }
    times = []
    for _ in range(3):
        t1 = time.time()
        r4_run(c, Xbig, Ybig, kbig)
        times.append(time.time() - t1)
    res = {
        "round": 4,
        "pre_registration": "DECISIONS.md D-038",
        "dev": dev,
        "winner": winner,
        "confirmation": per,
        "R1_passed": sum(x["R1_pass"] for x in per.values()),
        "R1_sets": len(per),
        "R1_all": all(x["R1_pass"] for x in per.values()),
        "R2_min_shuffle_ari": min(x["shuffle_ari"] for x in per.values()),
        "R3": r3,
        "R3_all": all(x["pass"] for x in r3.values()),
        "R4": {
            "rows": len(Ybig),
            "python_seconds_median": round(float(np.median(times)), 3),
            "R4_python_pass": bool(np.median(times) <= 2.0),
            "note": "language keys (script + langid) excluded; computed during encoding",
        },
    }
    res["R2_pass"] = res["R2_min_shuffle_ari"] >= 0.95
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", choices=["tiny", "small", "medium", "full"], default="small")
    ap.add_argument("--round", type=int, choices=[1, 2, 3, 4], default=1)
    a = ap.parse_args()
    if a.round in (2, 3, 4):
        out = {2: round2, 3: round3, 4: round4}[a.round](a.scale)
        p = ROOT / "bench" / "results" / f"refine-round{a.round}-{a.scale}.json"
        p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        log(f"wrote {p}  winner={out['winner']}")
        raise SystemExit(0)
    out = run(a.scale)
    p = ROOT / "bench" / "results" / f"refine-study-{a.scale}.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"wrote {p}  winner={out['winner']}")
