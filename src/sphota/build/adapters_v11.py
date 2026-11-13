# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Train and certify the v1.1 adapter set: nonlinear (ridge + residual MLP) adapters (D-028).

    python -m sphota.build.adapters_v11 --scale small

Same training pairs and held-out test as the frozen v1 adapters (``sphota.build.adapters``); only
the adapter form changes: ``y = x @ A + b + gelu(x @ W1 + c1) @ W2 + c2``, trained with torch,
shipped as numpy-only ``.npz`` (hidden weights in float16) and certified with the numpy forward
pass. Output: ``src/sphota/data/adapters/v1.1/`` (registry.json + files). The frozen v1 registry
is not touched.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from sphota.build import embed
from sphota.build.adapters import (
    MODELS,
    anchor_canonical,
    calibrate_temperature_scale,
    training_texts,
)
from sphota.build.common import ROOT, log
from sphota.build.release import load

OUT = ROOT / "src" / "sphota" / "data" / "adapters" / "v1.1"
# larger models, v1.1 only (no frozen v1 adapter exists for them; D-046)
EXTRA_MODELS: dict[str, dict] = {
    "intfloat/multilingual-e5-base": {
        "short": "me5-base",
        "license": "mit",
        "multilingual": True,
        "n_train": 40000,
        "onnx": {"repo": "Xenova/multilingual-e5-base", "pooling": "mean", "max_length": 512},
    },
    "BAAI/bge-base-en-v1.5": {
        "short": "bge-base-en",
        "license": "mit",
        "multilingual": False,
        "n_train": 40000,
        "onnx": {"repo": "Xenova/bge-base-en-v1.5", "pooling": "cls", "max_length": 512},
    },
}
ALL_MODELS = {**MODELS, **EXTRA_MODELS}
HIDDEN = 1024
EPOCHS = 40


def fit_mlp2(model: str, X: np.ndarray, Y: np.ndarray, meta: dict | None = None, seed: int = 0):
    """Ridge + residual GELU(tanh) MLP, trained with MSE; returns a numpy ``Adapter``."""
    import torch

    from sphota.adapters import Adapter
    from sphota.certify import fit_adapter

    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = fit_adapter(model, X, Y, meta=meta)
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    xt = torch.tensor(Xn, dtype=torch.float32, device=dev)
    bt = torch.tensor(Xn @ base.A + base.b, dtype=torch.float32, device=dev)
    yt = torch.tensor(Y, dtype=torch.float32, device=dev)
    l1 = torch.nn.Linear(X.shape[1], HIDDEN).to(dev)
    l2 = torch.nn.Linear(HIDDEN, Y.shape[1]).to(dev)
    torch.nn.init.zeros_(l2.weight)
    torch.nn.init.zeros_(l2.bias)
    act = torch.nn.GELU(approximate="tanh")
    params = [*l1.parameters(), *l2.parameters()]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    var = float(yt.var(0).sum())
    for _ in range(EPOCHS):
        perm = torch.randperm(len(xt), device=dev)
        for s in range(0, len(xt), 256):
            i = perm[s : s + 256]
            pred = bt[i] + l2(act(l1(xt[i])))
            loss = ((pred - yt[i]) ** 2).sum(1).mean() / var
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    with torch.no_grad():
        W1 = l1.weight.T.cpu().numpy().astype(np.float16)
        c1 = l1.bias.cpu().numpy().astype(np.float16)
        W2 = l2.weight.T.cpu().numpy().astype(np.float16)
        c2 = l2.bias.cpu().numpy().astype(np.float32)
    m = {**base.meta, "kind": "mlp2", "set": "v1.1", "hidden": HIDDEN, "activation": "gelu_tanh"}
    m["train"] = {
        **base.meta.get("train", {}),
        "method": "ridge + residual MLP (MSE)",
        "epochs": EPOCHS,
    }
    return Adapter(model, base.A, base.b, m, W1=W1, c1=c1, W2=W2, c2=c2)


def recall_at_10(Yq: np.ndarray, Yd: np.ndarray) -> float:
    q = Yq / np.linalg.norm(Yq, axis=1, keepdims=True)
    d = Yd / np.linalg.norm(Yd, axis=1, keepdims=True)
    top = np.argsort(-(q @ d.T), axis=1)[:, :10]
    return float((top == np.arange(len(q))[:, None]).any(1).mean())


def run(scale: str, models: list[str]) -> dict:
    from sphota.adapters import get_adapter, load_adapter
    from sphota.certify import evaluate, grade
    from sphota.release import load_release, sha256_file
    from sphota.space import project

    rel = load_release()
    langs, _dev, test, _wiki, _tat = load(scale)
    OUT.mkdir(parents=True, exist_ok=True)
    reg_path = OUT / "registry.json"
    registry = (
        json.loads(reg_path.read_text(encoding="utf-8"))
        if reg_path.exists()
        else {"space": "s1", "set": "v1.1", "adapters": {}}
    )
    results = {}
    for model in models:
        cfg = ALL_MODELS[model]
        t0 = time.time()
        texts = training_texts(scale, cfg["n_train"])
        Y = anchor_canonical(scale, texts)
        X = embed.cached(model, f"adapter-train-{len(texts)}", texts, scale)
        n_test = cfg.get("n_test")
        tt = test.iloc[:n_test] if n_test else test
        suffix = f"_{n_test}" if n_test else ""
        test_X = {
            lg: embed.cached(model, f"flores_devtest_{lg}{suffix}", tt[lg].tolist(), scale)
            for lg in langs
        }
        Ya = {
            lg: project(
                rel,
                embed.cached(
                    "intfloat/multilingual-e5-small",
                    f"flores_devtest_{lg}{suffix}",
                    tt[lg].tolist(),
                    scale,
                ),
            )
            for lg in langs
        }
        meta = {
            "model_license": cfg["license"],
            "multilingual": cfg["multilingual"],
            "prefix": embed.PREFIX.get(model, ""),
            "train": {
                "pairs": len(texts),
                "data": ["wikipedia-20231101 (12 langs)", "flores200-dev"],
            },
        }
        ts = calibrate_temperature_scale(rel, X, Y, fit=lambda a, b: fit_mlp2("tmp", a, b))
        ad = fit_mlp2(model, X, Y, meta={**meta, "temperature_scale": ts})
        fname = f"{cfg['short']}.npz"
        p = ad.save(OUT / fname)
        ad = load_adapter(p)  # certify exactly what ships (float16 weights, numpy forward)
        m = evaluate({lg: ad(test_X[lg]) for lg in langs}, Ya)
        g = grade(m)
        try:
            v1_r10: float | None = recall_at_10(get_adapter(model, "v1")(test_X["en"]), Ya["en"])
        except Exception:  # models without a frozen v1 adapter (D-046)
            v1_r10 = None
        r10 = {"v1": v1_r10, "v1.1": recall_at_10(ad(test_X["en"]), Ya["en"])}
        cert = {
            k: m[k]
            for k in [
                "agreement_same_text_en",
                "agreement_same_text",
                "l1_top3_en",
                "l1_neighborhood_en",
                "l1_top3_all",
                "agreement_cross_lingual",
                "fidelity_en",
                "fidelity_all",
                "n_per_lang",
            ]
        }
        cert["grade"] = g["grade"]
        cert["retrieval_r10_en"] = round(r10["v1.1"], 4)
        registry["adapters"][model] = {
            "short": cfg["short"],
            "file": fname,
            "sha256": sha256_file(p),
            "kind": "mlp2",
            "model_license": cfg["license"],
            "multilingual": cfg["multilingual"],
            "prefix": ad.prefix,
            **({"onnx": cfg["onnx"]} if "onnx" in cfg else {}),
            "temperature_scale": ts,
            "certificate": cert,
        }
        reg_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        v1c = json.loads((rel.root / "adapters" / "registry.json").read_text(encoding="utf-8"))
        old = v1c["adapters"].get(model, {}).get("certificate")
        results[model] = {
            "v1.1": cert,
            "v1": old,
            "retrieval_r10_en": r10,
            "seconds": round(time.time() - t0),
        }
        log(
            f"{model}: L1 en "
            f"{(old['agreement_same_text_en'][0] if old else float('nan')):.3f}→{cert['agreement_same_text_en'][0]:.3f} "
            f"all {cert['agreement_same_text'][0]:.3f} r@10 {r10['v1.1']:.3f} {g['grade']}"
        )
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", choices=["tiny", "small", "medium", "full"], default="small")
    # the five registry models (bge-m3 has no certified v1 adapter to compare with)
    ap.add_argument("--models", nargs="*", default=[m for m in MODELS if m != "BAAI/bge-m3"])
    a = ap.parse_args()
    res = run(a.scale, a.models)
    out = ROOT / "bench" / "results" / f"adapters-v1.1-{a.scale}.json"
    if out.exists():  # merge: keep models trained in earlier runs
        res = {**json.loads(out.read_text(encoding="utf-8")), **res}
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"wrote {out}")
