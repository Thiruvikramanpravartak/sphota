# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Train, certify and register adapters for popular open models (milestone M4).

    python -m sphota.build.adapters --scale small [--models ...]

Training pairs: the same texts embedded by the anchor (→ canonical space) and by the model.
Texts: a Wikipedia sample (12 languages) + FLORES-200 dev. Test: FLORES-200 devtest (never
used for training), 12 languages. Results → registry.json + bench/results/adapters-<scale>.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sphota.build import embed
from sphota.build.common import ROOT, log
from sphota.build.release import MODEL_WEIGHT, REFERENCE_MODELS, load
from sphota.build.release import embeddings as anchor_embeddings
from sphota.certify import evaluate, fit_adapter, grade

MODELS: dict[str, dict] = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "short": "minilm-l6",
        "license": "apache-2.0",
        "multilingual": False,
        "n_train": 40000,
        "onnx": {"repo": "Xenova/all-MiniLM-L6-v2", "pooling": "mean", "max_length": 256},
    },
    "BAAI/bge-small-en-v1.5": {
        "short": "bge-small-en",
        "license": "mit",
        "multilingual": False,
        "n_train": 40000,
        "onnx": {"repo": "Xenova/bge-small-en-v1.5", "pooling": "cls", "max_length": 512},
    },
    "thenlper/gte-small": {
        "short": "gte-small",
        "license": "mit",
        "multilingual": False,
        "n_train": 40000,
        "onnx": {"repo": "Xenova/gte-small", "pooling": "mean", "max_length": 512},
    },
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
        "short": "pm-minilm-l12",
        "license": "apache-2.0",
        "multilingual": True,
        "n_train": 40000,
        "onnx": {
            "repo": "Xenova/paraphrase-multilingual-MiniLM-L12-v2",
            "pooling": "mean",
            "max_length": 128,
        },
    },
    "sentence-transformers/LaBSE": {
        "short": "labse",
        "license": "apache-2.0",
        "multilingual": True,
        "n_train": 20000,
    },
    "BAAI/bge-m3": {
        "short": "bge-m3",
        "license": "mit",
        "multilingual": True,
        "n_train": 6000,
        "n_test": 300,
    },
}


def training_texts(scale: str, n: int) -> list[str]:
    langs, dev, _test, wiki, _tat = load(scale)
    rng = np.random.default_rng(11)
    flores_dev = [t for lg in langs for t in dev[lg].tolist()]
    k = max(0, n - len(flores_dev))
    idx = np.sort(rng.choice(len(wiki), min(k, len(wiki)), replace=False))
    wiki_txt = wiki["text"].iloc[idx].tolist()
    out = wiki_txt + flores_dev
    if len(out) > n:  # small budgets: keep a balanced mix
        sel = np.sort(rng.choice(len(out), n, replace=False))
        out = [out[i] for i in sel]
    return out


def anchor_canonical(scale: str, texts: list[str]) -> np.ndarray:
    """Canonical vectors of ``texts`` via cached anchor embeddings (encode if missing)."""
    from sphota.release import load_release
    from sphota.space import project

    X = embed.cached("intfloat/multilingual-e5-small", f"adapter-train-{len(texts)}", texts, scale)
    return project(load_release(), X)


def run(scale: str, models: list[str], out_dir: Path, mlp: bool = False) -> dict:
    from sphota.release import load_release
    from sphota.space import project

    rel = load_release()
    langs, _dev, test, _wiki, _tat = load(scale)
    anchor_embeddings(scale)  # ensure cached
    results: dict = {}
    reg_path = out_dir / "registry.json"
    registry = (
        json.loads(reg_path.read_text(encoding="utf-8"))
        if reg_path.exists()
        else {"space": "s1", "adapters": {}}
    )
    for model in models:
        cfg = MODELS[model]
        t0 = time.time()
        texts = training_texts(scale, cfg["n_train"])
        Y = anchor_canonical(scale, texts)
        X = embed.cached(model, f"adapter-train-{len(texts)}", texts, scale)
        n_test = cfg.get("n_test")
        tt = test.iloc[:n_test] if n_test else test
        Ym, Ya = {}, {}
        test_X = {
            lg: embed.cached(
                model,
                f"flores_devtest_{lg}" + (f"_{n_test}" if n_test else ""),
                tt[lg].tolist(),
                scale,
            )
            for lg in langs
        }
        anchor_X = {
            lg: embed.cached(
                "intfloat/multilingual-e5-small",
                f"flores_devtest_{lg}" + (f"_{n_test}" if n_test else ""),
                tt[lg].tolist(),
                scale,
            )
            for lg in langs
        }
        ts = calibrate_temperature_scale(rel, X, Y)
        ad = fit_adapter(
            model,
            X,
            Y,
            alpha=1.0,
            prefix=embed.PREFIX.get(model, ""),
            meta={
                "model_license": cfg["license"],
                "multilingual": cfg["multilingual"],
                "temperature_scale": ts,
                "reference_model": MODEL_WEIGHT > 0 and model in REFERENCE_MODELS,
                "train": {
                    "pairs": len(texts),
                    "method": "ridge",
                    "alpha": 1.0,
                    "data": ["wikipedia-20231101 (12 langs)", "flores200-dev"],
                },
            },
        )
        for lg in langs:
            Ym[lg] = ad(test_X[lg])
            Ya[lg] = project(rel, anchor_X[lg])
        m = evaluate(Ym, Ya)
        g = grade(m)
        res = {
            "model": model,
            **m,
            **g,
            "reference_model": MODEL_WEIGHT > 0 and model in REFERENCE_MODELS,
            "temperature_scale": ts,
            "train_pairs": len(texts),
            "seconds": round(time.time() - t0),
        }
        if mlp:
            res["mlp"] = train_mlp_eval(model, X, Y, test_X, Ya)
        fname = f"{cfg['short']}.npz"
        p = ad.save(out_dir / fname)
        from sphota.release import sha256_file

        registry["adapters"][model] = {
            "short": cfg["short"],
            "file": fname,
            "sha256": sha256_file(p),
            "model_license": cfg["license"],
            "multilingual": cfg["multilingual"],
            "prefix": ad.prefix,
            **({"onnx": cfg["onnx"]} if "onnx" in cfg else {}),
            "certificate": {
                k: res[k]
                for k in [
                    "agreement_same_text_en",
                    "agreement_same_text",
                    "l1_top3_en",
                    "l1_neighborhood_en",
                    "l1_top3_all",
                    "agreement_cross_lingual",
                    "fidelity_en",
                    "fidelity_all",
                    "grade",
                    "n_per_lang",
                ]
            },
        }
        reg_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        results[model] = res
        log(
            f"{model}: en {m['agreement_same_text_en']} all {m['agreement_same_text']} "
            f"xl {m['agreement_cross_lingual']} fid {m['fidelity_en']}/{m['fidelity_all']} → {g['grade']}"
        )
    return results


def calibrate_temperature_scale(rel, X: np.ndarray, Y: np.ndarray) -> float:
    """Fit the adapter on 85% of the pairs; choose the temperature scale that maximizes the
    likelihood of the anchor's L1/L2 symbols on the held-out 15% (same procedure as the
    space's own calibration, D-007)."""
    from sphota.space import quantize

    n = len(X)
    cut = int(0.85 * n)
    ad = fit_adapter("tmp", X[:cut], Y[:cut])
    Yh, Ya = ad(X[cut:]), Y[cut:]
    target = quantize(rel, Ya, levels=2).indices
    best, best_ll = 1.0, -np.inf
    for s in np.exp(np.linspace(np.log(0.5), np.log(8), 25)):
        q = quantize(rel, Yh, levels=2, temp_scale=float(s))
        ll = 0.0
        for lvl in range(2):
            same = np.all(q.indices[:, :lvl] == target[:, :lvl], 1)
            hit = q.cand[lvl] == target[:, lvl][:, None]
            p = np.where(hit.any(1), (q.cand_p[lvl] * hit).sum(1), 1e-4)
            ll += float(np.log(np.maximum(p, 1e-4))[same].sum())
        if ll > best_ll:
            best, best_ll = float(s), ll
    return round(best, 4)


def train_mlp_eval(model, X, Y, test_X, Ya) -> dict:
    """Optional: one residual layer on top of the ridge map (torch), evaluated the same way."""
    import torch

    from sphota.adapters import Adapter

    ad0 = fit_adapter(model, X, Y)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    H = torch.tensor(Xn @ ad0.A + ad0.b)
    T = torch.tensor(Y)
    A2 = torch.zeros(Y.shape[1], Y.shape[1], requires_grad=True)
    b2 = torch.zeros(Y.shape[1], requires_grad=True)
    opt = torch.optim.Adam([A2, b2], lr=1e-3, weight_decay=1e-4)
    for _ in range(30):
        perm = torch.randperm(len(H))
        for s in range(0, len(H), 512):
            i = perm[s : s + 512]
            pred = H[i] + torch.relu(H[i]) @ A2 + b2
            loss = ((pred - T[i]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    ad = Adapter(model, ad0.A, ad0.b, {"kind": "mlp"}, A2.detach().numpy(), b2.detach().numpy())
    m = evaluate({lg: ad(v) for lg, v in test_X.items()}, Ya)
    return {k: m[k] for k in ["agreement_same_text_en", "agreement_same_text", "fidelity_en"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="small")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--mlp", action="store_true")
    a = ap.parse_args()
    out_dir = ROOT / "src" / "sphota" / "data" / "v1" / "adapters"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = run(a.scale, a.models, out_dir, mlp=a.mlp)
    p = ROOT / "bench" / "results" / f"adapters-{a.scale}.json"
    old = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    old.update(res)
    p.write_text(json.dumps(old, indent=2), encoding="utf-8")
    log(f"wrote {p}")


if __name__ == "__main__":
    main()
