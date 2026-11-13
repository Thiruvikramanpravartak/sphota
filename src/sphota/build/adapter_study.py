# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Adapter study: how much of the cross-model gap is the adapter? (D-028)

    python -m sphota.build.adapter_study --scale small

The decision gate failed on cross-model agreement (D-022): codes from other embedding models,
mapped into the frozen s1 space by linear (ridge) adapters, land in the anchor's cell too rarely.
This study holds everything else fixed (frozen s1 space, cached embeddings of the same training
texts, held-out FLORES-200 devtest) and varies only the adapter:

* ``procrustes``: rotation + scale only (lower bound on capacity);
* ``ridge``: the shipped adapter;
* ``mlp``: ridge + a residual two-layer network, trained with MSE;
* ``mlp+code``: the same network, also trained to put the vector in the anchor's own cell
  (cross-entropy over the L1 codebook with the release's calibrated temperature): an adapter
  that optimizes the code, not just the vector.

Metrics (same text, model vs anchor): L1 cell agreement on the frozen 4096-cell codebook, region
agreement (32 regions) and 256-cell agreement (the gate's granularity), for English and for all
12 FLORES languages (multilingual models only), plus a data-scaling curve for ridge vs mlp+code.
Writes ``bench/results/adapter-study-<scale>.json``.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from sphota.build import embed
from sphota.build.common import ROOT, log

MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": {"n": 40000, "multilingual": False},
    "BAAI/bge-small-en-v1.5": {"n": 40000, "multilingual": False},
    "thenlper/gte-small": {"n": 40000, "multilingual": False},
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
        "n": 40000,
        "multilingual": True,
    },
    "sentence-transformers/LaBSE": {"n": 20000, "multilingual": True},
}
# larger models (D-043): does the nonlinear-adapter gain hold beyond small models?
MORE_MODELS = {
    "intfloat/multilingual-e5-base": {"n": 40000, "multilingual": True},
    "BAAI/bge-base-en-v1.5": {"n": 40000, "multilingual": False},
    "sentence-transformers/distiluse-base-multilingual-cased-v2": {
        "n": 40000,
        "multilingual": True,
    },
    "mixedbread-ai/mxbai-embed-large-v1": {"n": 40000, "multilingual": False},
}
MODELS_ALL = {**MODELS, **MORE_MODELS}
ANCHOR = "intfloat/multilingual-e5-small"


def l2n(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def nearest(Y: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Index of the nearest centroid (squared Euclidean), batched."""
    cn = (C**2).sum(1)
    out = np.empty(len(Y), dtype=np.int64)
    for s in range(0, len(Y), 4096):
        y = Y[s : s + 4096]
        out[s : s + 4096] = np.argmin(cn[None] - 2 * y @ C.T, axis=1)
    return out


# ------------------------------------------------------------------ adapters


def fit_ridge(X, Y, alpha=1.0):
    from sphota.certify import fit_adapter

    ad = fit_adapter("m", X, Y, alpha=alpha)
    return lambda Z: l2n(Z) @ ad.A + ad.b


def fit_procrustes(X, Y):
    Xn = l2n(X)
    mx, my = Xn.mean(0), Y.mean(0)
    Xc, Yc = Xn - mx, Y - my
    d = min(Xc.shape[1], Yc.shape[1])
    # project to a common width first (ridge to d dims), then the best rotation + scale
    U, S, Vt = np.linalg.svd(Xc.T @ Yc, full_matrices=False)
    R = U @ Vt  # (dx, dy) orthonormal columns
    s = S.sum() / max((Xc**2).sum(), 1e-12)
    _ = d
    return lambda Z: (l2n(Z) - mx) @ R * s + my


def fit_mlp(X, Y, C1=None, T=None, lam=0.0, epochs=40, hidden=1024, seed=0, levels=None):
    """Ridge + residual MLP; with ``lam > 0`` also cross-entropy towards the anchor's L1 cell.

    ``levels``: list of (centroids, temperature, weight) for a multi-level code loss (overrides
    C1/T/lam): the vector is pulled into the anchor's cell at every listed granularity.
    """
    if levels is None:
        levels = [(C1, T, lam)] if lam > 0 else []
    import torch

    torch.manual_seed(seed)
    base = fit_ridge(X, Y)
    Xn = torch.tensor(l2n(X), dtype=torch.float32)
    B = torch.tensor(base(X), dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)
    net = torch.nn.Sequential(
        torch.nn.Linear(Xn.shape[1], hidden),
        torch.nn.GELU(),
        torch.nn.Linear(hidden, Y.shape[1]),
    )
    torch.nn.init.zeros_(net[2].weight)
    torch.nn.init.zeros_(net[2].bias)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    heads = [
        (torch.tensor(C, dtype=torch.float32), float(t), float(w), torch.tensor(nearest(Y, C)))
        for C, t, w in levels
    ]
    var = float(Yt.var(0).sum())
    for _ in range(epochs):
        perm = torch.randperm(len(Xn))
        for s in range(0, len(Xn), 256):
            i = perm[s : s + 256]
            pred = B[i] + net(Xn[i])
            loss = ((pred - Yt[i]) ** 2).sum(1).mean() / var
            for Ct, t, w, target in heads:
                d2 = (Ct**2).sum(1)[None] - 2 * pred @ Ct.T + (pred**2).sum(1, keepdim=True)
                loss = loss + w * torch.nn.functional.cross_entropy(-d2 / t, target[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    net.eval()

    def apply(Z):
        with torch.no_grad():
            z = torch.tensor(l2n(Z), dtype=torch.float32)
            return (torch.tensor(base(Z), dtype=torch.float32) + net(z)).numpy()

    return apply


# ------------------------------------------------------------------ study


def run(scale: str, models: list[str], quick: bool = False) -> dict:
    from sphota.build.adapters import anchor_canonical, training_texts
    from sphota.build.fit import kmeans
    from sphota.build.release import load
    from sphota.regions import _load as regions
    from sphota.release import load_release
    from sphota.space import project

    rel = load_release()
    C1 = rel.codebooks[0].astype(np.float64)
    T1 = float(rel.temperatures[0])
    CR, TR, _ = regions()
    langs, _dev, test, _wiki, _tat = load(scale)
    Ya = {
        lg: project(rel, embed.cached(ANCHOR, f"flores_devtest_{lg}", test[lg].tolist(), scale))
        for lg in langs
    }
    # the gate's granularity: 256 cells re-clustered on the anchor's canonical training vectors
    tr40 = training_texts(scale, 40000)
    Y40 = anchor_canonical(scale, tr40)
    C256 = kmeans(Y40, 256, seed=0, iters=30)
    grids = {"l1_4096": C1, "cells_256": C256, "regions_32": CR}
    anchor_idx = {g: {lg: nearest(Ya[lg], C) for lg in langs} for g, C in grids.items()}

    def score(adapt, model, multilingual):
        out = {}
        for g, C in grids.items():
            per = {}
            for lg in langs if multilingual else ["en"]:
                Xt = embed.cached(model, f"flores_devtest_{lg}", test[lg].tolist(), scale)
                per[lg] = float((nearest(adapt(Xt), C) == anchor_idx[g][lg]).mean())
            out[g] = {
                "en": round(per["en"], 4),
                "all": round(float(np.mean(list(per.values()))), 4),
            }
        return out

    res: dict = {"scale": scale, "anchor": ANCHOR, "space": rel.version, "models": {}}
    for model in models:
        cfg = MODELS_ALL[model]
        texts = training_texts(scale, cfg["n"])
        X = embed.cached(model, f"adapter-train-{cfg['n']}", texts, scale)
        Y = Y40 if cfg["n"] == 40000 else anchor_canonical(scale, texts)
        m: dict = {}
        variants = {
            "procrustes": lambda: fit_procrustes(X, Y),
            "ridge": lambda: fit_ridge(X, Y),
            "mlp": lambda: fit_mlp(X, Y, epochs=10 if quick else 40),
            "mlp+code": lambda: fit_mlp(X, Y, C1, T1, lam=0.5, epochs=10 if quick else 40),
            # whole hierarchy: regions (32) + L1 (4096), so coarse agreement is not traded away
            "mlp+hier": lambda: fit_mlp(
                X,
                Y,
                epochs=10 if quick else 40,
                levels=[(CR, TR, 0.5), (C1, T1, 0.5)],
            ),
        }
        for name, make in variants.items():
            t0 = time.time()
            m[name] = score(make(), model, cfg["multilingual"])
            m[name]["train_s"] = round(time.time() - t0, 1)
            log(f"{model.split('/')[-1]:<40} {name:<11} {m[name]}")
        res["models"][model] = m
    # data scaling (one English and one multilingual model)
    res["scaling"] = {}
    for model in [
        x
        for x in models
        if x
        in (
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
    ]:
        X = embed.cached(model, "adapter-train-40000", tr40, scale)
        rng = np.random.default_rng(0)
        curve = {}
        for n in [2500, 5000, 10000, 20000, 40000]:
            idx = rng.permutation(len(X))[:n]
            curve[n] = {
                "ridge": score(fit_ridge(X[idx], Y40[idx]), model, True)["l1_4096"]["all"],
                "mlp+hier": score(
                    fit_mlp(
                        X[idx],
                        Y40[idx],
                        epochs=10 if quick else 40,
                        levels=[(CR, TR, 0.5), (C1, T1, 0.5)],
                    ),
                    model,
                    True,
                )["l1_4096"]["all"],
            }
            log(f"scaling {model.split('/')[-1]} n={n}: {curve[n]}")
        res["scaling"][model] = curve
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", choices=["tiny", "small", "medium", "full"], default="small")
    ap.add_argument("--models", nargs="*", default=list(MODELS), help=f"any of {list(MODELS_ALL)}")
    ap.add_argument("--quick", action="store_true", help="10 epochs instead of 40")
    a = ap.parse_args()
    out = run(a.scale, a.models, a.quick)
    p = ROOT / "bench" / "results" / f"adapter-study-{a.scale}.json"
    if p.exists() and set(a.models) != set(MODELS):  # a subset: merge, keep earlier models
        old = json.loads(p.read_text(encoding="utf-8"))
        old["models"].update(out["models"])
        old.setdefault("scaling", {}).update(out.get("scaling", {}))
        out = old
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"wrote {p}")
