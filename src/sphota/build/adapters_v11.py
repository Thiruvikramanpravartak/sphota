# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Train and certify the nonlinear adapter sets (ridge + residual MLP): v1.1 (D-028) and v1.2
(the same recipe plus short-command training pairs, D-052).

    python -m sphota.build.adapters_v11 --scale small               # v1.1
    python -m sphota.build.adapters_v11 --scale small --set v1.2    # v1.2 (+ held-out command eval)

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
CMD_PER_LANG = 2000  # v1.2: non-English command pairs per language (multilingual models only)


def command_texts(scale: str, multilingual: bool) -> list[str]:
    """v1.2 command pairs (D-052): MASSIVE train (test texts excluded), every English utterance,
    plus a fixed sample of CMD_PER_LANG per other language for multilingual models."""
    from sphota.build.gate import massive_texts

    langs = load(scale)[0]
    T = massive_texts(langs, "train", scale)
    out = list(T["en"])
    if multilingual:
        for j, lg in enumerate(langs):
            if lg == "en":
                continue
            rng = np.random.default_rng(100 + j)
            idx = np.sort(rng.choice(len(T[lg]), min(CMD_PER_LANG, len(T[lg])), replace=False))
            out += [T[lg][i] for i in idx]
    return out


def command_eval(scale: str, model: str, multilingual: bool, adapters: dict) -> dict:
    """Held-out MASSIVE test: share of commands whose adapted vector lands in the anchor's own
    L1 cell (4096) and region (32); English, and all languages for multilingual models."""
    from sphota.build.adapter_study import nearest
    from sphota.build.gate import massive_texts
    from sphota.regions import _load as regions
    from sphota.release import load_release
    from sphota.space import project

    rel = load_release()
    C1 = rel.codebooks[0].astype(np.float64)
    CR = regions()[0]
    langs = load(scale)[0]
    T = massive_texts(langs, "test", scale)
    out: dict = {}
    for lg in langs if multilingual else ["en"]:
        n = len(T[lg])
        ya = project(rel, embed.cached(ANCHOR, f"massive-test-{lg}-{n}", T[lg], scale))
        xm = embed.cached(model, f"massive-test-{lg}-{n}", T[lg], scale)
        a1, ar = nearest(ya, C1), nearest(ya, CR)
        for name, ad in adapters.items():
            y = ad(xm)
            out.setdefault(name, {})[lg] = {
                "l1": round(float((nearest(y, C1) == a1).mean()), 4),
                "region": round(float((nearest(y, CR) == ar).mean()), 4),
            }
    summ = {}
    for name, per in out.items():
        summ[name] = {
            "l1_en": per["en"]["l1"],
            "region_en": per["en"]["region"],
            "l1_all": round(float(np.mean([v["l1"] for v in per.values()])), 4),
            "region_all": round(float(np.mean([v["region"] for v in per.values()])), 4),
            "per_lang": per,
        }
    return summ


ANCHOR = "intfloat/multilingual-e5-small"


def fit_mlp2(
    model: str,
    X: np.ndarray,
    Y: np.ndarray,
    meta: dict | None = None,
    seed: int = 0,
    hidden: int = HIDDEN,
    set_name: str = "v1.1",
):
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
    l1 = torch.nn.Linear(X.shape[1], hidden).to(dev)
    l2 = torch.nn.Linear(hidden, Y.shape[1]).to(dev)
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
    m = {**base.meta, "kind": "mlp2", "set": set_name, "hidden": hidden, "activation": "gelu_tanh"}
    m["train"] = {
        **base.meta.get("train", {}),
        "method": "ridge + residual MLP (MSE)"
        + (" + short-command pairs (D-052)" if set_name == "v1.2" else ""),
        "epochs": EPOCHS,
    }
    return Adapter(model, base.A, base.b, m, W1=W1, c1=c1, W2=W2, c2=c2)


def recall_at_10(Yq: np.ndarray, Yd: np.ndarray) -> float:
    q = Yq / np.linalg.norm(Yq, axis=1, keepdims=True)
    d = Yd / np.linalg.norm(Yd, axis=1, keepdims=True)
    top = np.argsort(-(q @ d.T), axis=1)[:, :10]
    return float((top == np.arange(len(q))[:, None]).any(1).mean())


def run(scale: str, models: list[str], set_name: str = "v1.1") -> dict:
    from sphota.adapters import get_adapter, load_adapter
    from sphota.certify import evaluate, grade
    from sphota.release import load_release, sha256_file
    from sphota.space import project

    rel = load_release()
    langs, _dev, test, _wiki, _tat = load(scale)
    commands = set_name == "v1.2"
    out_dir = OUT.parent / set_name
    out_dir.mkdir(parents=True, exist_ok=True)
    reg_path = out_dir / "registry.json"
    registry = (
        json.loads(reg_path.read_text(encoding="utf-8"))
        if reg_path.exists()
        else {"space": "s1", "set": set_name, "adapters": {}}
    )
    results = {}
    for model in models:
        cfg = ALL_MODELS[model]
        t0 = time.time()
        texts = training_texts(scale, cfg["n_train"])
        Y = anchor_canonical(scale, texts)
        X = embed.cached(model, f"adapter-train-{len(texts)}", texts, scale)
        data = ["wikipedia-20231101 (12 langs)", "flores200-dev"]
        if commands:  # D-052: + short commands (MASSIVE train, CC BY 4.0; test texts excluded)
            ct = command_texts(scale, cfg["multilingual"])
            tag = f"cmd-train-{'ml' if cfg['multilingual'] else 'en'}-{len(ct)}"
            X = np.concatenate([X, embed.cached(model, tag, ct, scale)])
            Y = np.concatenate([Y, project(rel, embed.cached(ANCHOR, tag, ct, scale))])
            data.append(f"amazon-massive-train ({len(ct)} commands)")
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
            "train": {"pairs": len(X), "data": data},
        }
        ts = calibrate_temperature_scale(rel, X, Y, fit=lambda a, b: fit_mlp2("tmp", a, b))
        ad = fit_mlp2(model, X, Y, meta={**meta, "temperature_scale": ts}, set_name=set_name)
        fname = f"{cfg['short']}.npz"
        p = ad.save(out_dir / fname)
        ad = load_adapter(p)  # certify exactly what ships (float16 weights, numpy forward)
        m = evaluate({lg: ad(test_X[lg]) for lg in langs}, Ya)
        g = grade(m)
        try:
            v1_r10: float | None = recall_at_10(get_adapter(model, "v1")(test_X["en"]), Ya["en"])
        except Exception:  # models without a frozen v1 adapter (D-046)
            v1_r10 = None
        r10 = {"v1": v1_r10, set_name: recall_at_10(ad(test_X["en"]), Ya["en"])}
        if commands:
            r10["v1.1"] = recall_at_10(get_adapter(model, "v1.1")(test_X["en"]), Ya["en"])
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
        cert["retrieval_r10_en"] = round(r10[set_name], 4)
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
        if commands:
            v11 = json.loads((OUT / "registry.json").read_text(encoding="utf-8"))
            old = v11["adapters"].get(model, {}).get("certificate")
        results[model] = {
            set_name: cert,
            "v1.1" if commands else "v1": old,
            "retrieval_r10_en": r10,
            "seconds": round(time.time() - t0),
        }
        if commands:
            results[model]["commands"] = command_eval(
                scale, model, cfg["multilingual"], {"v1.1": get_adapter(model, "v1.1"), "v1.2": ad}
            )
        log(
            f"{model}: L1 en "
            f"{(old['agreement_same_text_en'][0] if old else float('nan')):.3f}→{cert['agreement_same_text_en'][0]:.3f} "
            f"all {cert['agreement_same_text'][0]:.3f} r@10 {r10[set_name]:.3f} {g['grade']}"
            + (f" commands {results[model]['commands']}" if commands else "")
        )
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", choices=["tiny", "small", "medium", "full"], default="small")
    # the five registry models (bge-m3 has no certified v1 adapter to compare with)
    ap.add_argument("--models", nargs="*", default=[m for m in MODELS if m != "BAAI/bge-m3"])
    ap.add_argument("--set", choices=["v1.1", "v1.2"], default="v1.1")
    a = ap.parse_args()
    res = run(a.scale, a.models, a.set)
    out = ROOT / "bench" / "results" / f"adapters-{a.set}-{a.scale}.json"
    if out.exists():  # merge: keep models trained in earlier runs
        res = {**json.loads(out.read_text(encoding="utf-8")), **res}
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"wrote {out}")
