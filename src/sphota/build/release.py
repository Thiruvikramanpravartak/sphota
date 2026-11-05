# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Build and freeze the Sphota v1 release (milestone M2).

    python -m sphota.build.release --scale small [--out src/sphota/data/v1]

Steps (each cached under data/<scale>/build/):
1. embed corpus + parallel data with the anchor model,
2. fit the canonical space (PCA + meaning-LDA alignment),
3. fit the residual k-means codebooks,
4. calibrate per-level temperatures and the stable-mode noise scale on held-out pairs,
5. build multilingual c-TF-IDF labels and the L1 neighbour table,
6. write artifacts + manifest with SHA-256 checksums.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sphota.build import embed
from sphota.build.common import DATA_DIR, LANGS, ROOT, SCALES, log
from sphota.build.fit import ResidualKMeans, cross_model_residuals, fit_space, prefix_agree

ANCHOR = "intfloat/multilingual-e5-small"
ANCHOR_PREFIX = "query: "
SPLIT_MEANING = 800  # FLORES dev meanings [0, 800) fit the space; [800, 997) calibrate
# Optional cross-model noise term (D-012). Measured: no gain at K1=4096, so s1 uses weight 0;
# the machinery is kept for full-scale experiments.
REFERENCE_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "BAAI/bge-small-en-v1.5",
]
MODEL_WEIGHT = 0.0


# ------------------------------------------------------------------ data


def load(scale: str):
    d = DATA_DIR / scale
    langs = LANGS[: SCALES[scale]["n_langs"]]
    dev = pd.read_parquet(d / "flores_dev.parquet")[langs]
    test = pd.read_parquet(d / "flores_devtest.parquet")[langs]
    wiki = pd.read_parquet(d / "wiki.parquet")
    tat = pd.read_parquet(d / "tatoeba.parquet")
    return langs, dev, test, wiki, tat


def embeddings(scale: str, model: str = ANCHOR):
    langs, dev, test, wiki, tat = load(scale)
    E: dict = {}
    for lang in langs:
        E[("dev", lang)] = embed.cached(model, f"flores_dev_{lang}", dev[lang].tolist(), scale)
        E[("test", lang)] = embed.cached(
            model, f"flores_devtest_{lang}", test[lang].tolist(), scale
        )
    E["wiki"] = embed.cached(model, "wiki", wiki["text"].tolist(), scale)
    E["tat_x"] = embed.cached(model, "tatoeba_x", tat["text"].tolist(), scale)
    E["tat_en"] = embed.cached(model, "tatoeba_en", tat["en"].tolist(), scale)
    return E


def tatoeba_split(n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    perm = rng.permutation(n)
    cut = int(0.7 * n)
    return np.sort(perm[:cut]), np.sort(perm[cut:])


# ------------------------------------------------------------------ calibration


def _level_cands(R: np.ndarray, C: np.ndarray, m: int = 16):
    cn = (C.astype(np.float64) ** 2).sum(1)
    d2 = (
        cn[None]
        - 2 * R.astype(np.float64) @ C.T.astype(np.float64)
        + (R.astype(np.float64) ** 2).sum(1, keepdims=True)
    )
    part = np.argpartition(d2, m - 1, 1)[:, :m]
    dd = np.take_along_axis(d2, part, 1)
    o = np.argsort(dd, 1)
    return np.take_along_axis(part, o, 1), np.take_along_axis(dd, o, 1)


def calibrate_temperatures(
    codebooks: list[np.ndarray], Ya: np.ndarray, Yb: np.ndarray
) -> list[float]:
    """Max-likelihood temperature per level: P(partner's symbol | my distances, same prefix)."""
    temps = []
    Ra, Rb = Ya.astype(np.float32).copy(), Yb.astype(np.float32).copy()
    same = np.ones(len(Ya), bool)
    for C in codebooks:
        ca, da = _level_cands(Ra, C)
        cb, db = _level_cands(Rb, C)
        ia, ib = ca[:, 0], cb[:, 0]
        rows = []
        for cand, d, tgt in [(ca, da, ib), (cb, db, ia)]:
            hit = cand == tgt[:, None]
            rows.append((d - d[:, :1], hit))
        best_T, best_ll = None, -np.inf
        span = float(np.median(rows[0][0][:, 1])) + 1e-9
        for T in span * np.logspace(-2, 1.5, 60):
            ll = 0.0
            for dd, hit in rows:
                logits = -dd / T
                logits -= logits.max(1, keepdims=True)
                logp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
                p_t = np.where(hit.any(1), (logp * hit).sum(1), np.log(1e-4))
                ll += float(p_t[same].sum())
            if ll > best_ll:
                best_T, best_ll = T, ll
        temps.append(float(best_T))
        same &= ia == ib
        Ra -= C[ia]
        Rb -= C[ib]
    return temps


def reliability(codebooks, temps, Ya, Yb, bins=10) -> list[dict]:
    """Reliability table per level: predicted confidence vs observed agreement (cumulative)."""
    from sphota.release import Release
    from sphota.space import quantize

    rel = Release(
        root=Path("."),
        manifest={},
        mu=np.zeros(1),
        W=np.zeros((1, 1)),
        codebooks=codebooks,
        temperatures=np.array(temps),
        noise_scale=1.0,
    )
    qa, qb = quantize(rel, Ya), quantize(rel, Yb)
    out = []
    for lvl in range(len(codebooks)):
        conf = np.concatenate([qa.confidence[:, lvl], qb.confidence[:, lvl]])
        agree = np.all(qa.indices[:, : lvl + 1] == qb.indices[:, : lvl + 1], 1)
        agree = np.concatenate([agree, agree])
        edges = np.linspace(0, 1, bins + 1)
        tab = []
        ece = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
            if m.sum():
                tab.append(
                    {
                        "bin": [round(lo, 2), round(hi, 2)],
                        "n": int(m.sum()),
                        "mean_conf": float(conf[m].mean()),
                        "agree": float(agree[m].mean()),
                    }
                )
                ece += m.mean() * abs(conf[m].mean() - agree[m].mean())
        out.append({"level": lvl + 1, "ece": float(ece), "bins": tab})
    return out


def calibrate_noise(codebooks, temps, Ya, Yb) -> tuple[float, list[dict]]:
    from sphota.release import Release
    from sphota.space import quantize

    rel = Release(
        root=Path("."),
        manifest={},
        mu=np.zeros(1),
        W=np.zeros((1, 1)),
        codebooks=codebooks,
        temperatures=np.array(temps),
        noise_scale=1.0,
    )
    base = prefix_agree(quantize(rel, Ya).indices, quantize(rel, Yb).indices)
    trials = [{"sigma": 0.0, "agree": base}]
    best = (0.0, sum(base))
    for s in [0.25, 0.5, 0.75, 1.0, 1.5]:
        rel.noise_scale = s
        a = prefix_agree(
            quantize(rel, Ya, stable=True).indices, quantize(rel, Yb, stable=True).indices
        )
        trials.append({"sigma": s, "agree": a})
        log(f"stable sigma={s}: {np.round(a, 3)} (plain {np.round(base, 3)})")
        if sum(a) > best[1]:
            best = (s, sum(a))
    return (best[0] if best[0] > 0 else 0.5), trials


# ------------------------------------------------------------------ main


def build(scale: str, out: Path, D: int, K: tuple[int, ...], reg: float, seed: int = 0) -> dict:
    langs, dev, test, wiki, tat = load(scale)
    E = embeddings(scale)
    tr, ca = tatoeba_split(len(tat))
    fit_ids = np.arange(SPLIT_MEANING)
    cal_ids = np.arange(SPLIT_MEANING, len(dev))
    groups = [np.stack([E[("dev", lg)][i] for lg in langs]) for i in fit_ids]
    groups += [np.stack([E["tat_x"][i], E["tat_en"][i]]) for i in tr]
    corpus = np.concatenate([E["wiki"]] + [E[("dev", lg)][fit_ids] for lg in langs])
    log(f"fit space: corpus={len(corpus)} groups={len(groups)} D={D}")
    resid = None
    if MODEL_WEIGHT > 0:
        from sphota.build.adapters import training_texts

        texts = training_texts(scale, 40000)
        XA = embed.cached(ANCHOR, f"adapter-train-{len(texts)}", texts, scale)
        others = [
            embed.cached(m, f"adapter-train-{len(texts)}", texts, scale) for m in REFERENCE_MODELS
        ]
        resid = cross_model_residuals(XA, others)
        log(f"cross-model residuals from {len(REFERENCE_MODELS)} reference models: {resid.shape}")
    space = fit_space(
        corpus,
        groups,
        D=D,
        P=min(D * 2, corpus.shape[1]),
        reg=reg,
        model_residuals=resid,
        model_weight=MODEL_WEIGHT,
    )
    Yc = space(corpus)
    log(f"fit residual k-means K={K}")
    q = ResidualKMeans(K, seed=seed, iters=40).fit(Yc)
    cbs = q.codebooks
    # calibration pairs: held-out FLORES dev meanings (en vs every language) + held-out Tatoeba
    Ya = np.concatenate(
        [space(E[("dev", "en")][cal_ids])] * (len(langs) - 1) + [space(E["tat_en"][ca])]
    )
    Yb = np.concatenate(
        [space(E[("dev", lg)][cal_ids]) for lg in langs[1:]] + [space(E["tat_x"][ca])]
    )
    temps = calibrate_temperatures(cbs, Ya, Yb)
    log(f"temperatures={np.round(temps, 3)}")
    rel_table = reliability(cbs, temps, Ya, Yb)
    log("ECE per level: " + ", ".join(f"{r['ece']:.3f}" for r in rel_table))
    sigma, stable_trials = calibrate_noise(cbs, temps, Ya, Yb)

    codes_wiki = q.encode(space(E["wiki"]))

    # neighbour table
    C1 = cbs[0].astype(np.float64)
    d2 = (C1**2).sum(1)[:, None] - 2 * C1 @ C1.T + (C1**2).sum(1)[None, :]
    np.fill_diagonal(d2, np.inf)
    nb = np.argsort(d2, 1)[:, :32].astype(np.int32)
    nbd = np.sqrt(np.maximum(np.take_along_axis(d2, nb, 1), 0)).astype(np.float32)
    counts = np.bincount(codes_wiki[:, 0], minlength=K[0])

    # ---- write
    out.mkdir(parents=True, exist_ok=True)
    (out / "adapters").mkdir(exist_ok=True)
    np.savez(out / "space.npz", mu=space.mu, W=space.W, eigvals=space.eigvals.astype(np.float32))
    np.savez(
        out / "codebooks.npz", **{f"level{i + 1}": c.astype(np.float32) for i, c in enumerate(cbs)}
    )
    np.savez(
        out / "neighbors.npz",
        l1_neighbors=nb,
        l1_neighbor_dist=nbd,
        l1_counts=counts.astype(np.int32),
    )
    (out / "calibration.json").write_text(
        json.dumps(
            {
                "temperatures": temps,
                "noise_scale": sigma,
                "top_m": 16,
                "reliability": rel_table,
                "stable_trials": stable_trials,
                "n_pairs": len(Ya),
            },
            indent=1,
        )
    )
    from sphota.build.labels import build_labels

    build_labels(scale, out)
    write_manifest(out, scale=scale, D=D, K=K, reg=reg, langs=langs, n_corpus=len(corpus))
    return {"temps": temps, "sigma": sigma}


def write_manifest(out: Path, **info) -> dict:
    from sphota.release import sha256_file

    files = {}
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "manifest.json" and "adapters" not in p.parts:
            files[str(p.relative_to(out))] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    manifest = {
        "name": "sphota",
        "version": "s1",
        "status": f"frozen ({info['scale']} scale build)",
        "created": dt.date.today().isoformat(),
        "anchor": {
            "model": ANCHOR,
            "prefix": ANCHOR_PREFIX,
            "pooling": "mean",
            "normalize": True,
            "reference_onnx": "intfloat/multilingual-e5-small:onnx/model.onnx",
        },
        "space": {
            "dim": info["D"],
            "alignment": "meaning-LDA (languages + models)",
            "reg": info["reg"],
            "reference_models": REFERENCE_MODELS,
            "model_weight": MODEL_WEIGHT,
            "transform": "y = (normalize(x) - mu) @ W",
        },
        "quantizer": {
            "type": "residual-kmeans",
            "sizes": list(info["K"]),
            "assignment": "nearest centroid (squared L2), ties -> lowest index",
        },
        "code_format": {
            "alphabet": "crockford-base32",
            "widths": [4, 2, 2, 2],
            "perm_a": [0x5BD1F, 0x2A5, 0x1B3, 0x3C7],
            "perm_b": [0x3A6E9, 0x15D, 0x0C9, 0x2E1],
        },
        "training_data": {
            "languages": info["langs"],
            "corpus_size": info["n_corpus"],
            "sources": ["wikipedia-20231101", "flores200-dev", "tatoeba"],
        },
        "files": files,
        "license": "Apache-2.0 (artifacts derived from CC-BY-SA data; see NOTICE)",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def refresh_manifest(out: Path) -> dict:
    """Recompute file checksums, keeping all other manifest fields."""
    from sphota.release import sha256_file

    mpath = out / "manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    manifest["files"] = {
        str(p.relative_to(out)): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "manifest.json" and "adapters" not in p.parts
    }
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="small", choices=list(SCALES))
    ap.add_argument("--out", default=str(ROOT / "src" / "sphota" / "data" / "v1"))
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--reg", type=float, default=0.05)
    ap.add_argument("--k", type=int, nargs=4, default=None)
    a = ap.parse_args()
    K = tuple(a.k) if a.k else SCALES[a.scale]["k"]
    build(a.scale, Path(a.out), a.dim, K, a.reg)


if __name__ == "__main__":
    main()
