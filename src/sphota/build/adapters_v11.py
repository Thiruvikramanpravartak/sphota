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


def command_texts(scale: str, multilingual: bool, per_lang: int | None = CMD_PER_LANG) -> list[str]:
    """Command pairs: MASSIVE train (test texts excluded), every English utterance, plus for
    multilingual models a fixed sample of ``per_lang`` per other language (v1.2, D-052) or all
    of them (``per_lang=None``: v1.3 and run 4, D-055/D-056)."""
    from sphota.build.gate import massive_texts

    langs = load(scale)[0]
    T = massive_texts(langs, "train", scale)
    out = list(T["en"])
    if multilingual:
        for j, lg in enumerate(langs):
            if lg == "en":
                continue
            if per_lang is None:
                out += list(T[lg])
                continue
            rng = np.random.default_rng(100 + j)
            idx = np.sort(rng.choice(len(T[lg]), min(per_lang, len(T[lg])), replace=False))
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
        + (" + short-command pairs (D-052)" if set_name in ("v1.2", "v1.3") else ""),
        "epochs": EPOCHS,
    }
    return Adapter(model, base.A, base.b, m, W1=W1, c1=c1, W2=W2, c2=c2)


def recall_at_10(Yq: np.ndarray, Yd: np.ndarray) -> float:
    q = Yq / np.linalg.norm(Yq, axis=1, keepdims=True)
    d = Yd / np.linalg.norm(Yd, axis=1, keepdims=True)
    top = np.argsort(-(q @ d.T), axis=1)[:, :10]
    return float((top == np.arange(len(q))[:, None]).any(1).mean())


# adapter sets: the set each is compared with, command pairs per non-English language
# (False: none; an int: a fixed sample; None: all), and an optional training-size override
SETS: dict[str, dict] = {
    "v1.1": {"prev": "v1", "cmd_per_lang": False, "n_train": None},  # D-028
    "v1.2": {"prev": "v1.1", "cmd_per_lang": CMD_PER_LANG, "n_train": None},  # D-052
    "v1.3": {"prev": "v1.2", "cmd_per_lang": None, "n_train": 400000},  # D-056 (GPU, medium)
}
CERT_KEYS = [
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


def _test_sets(scale: str, model: str, cfg: dict):
    from sphota.release import load_release
    from sphota.space import project

    rel = load_release()
    langs, _dev, test, _wiki, _tat = load(scale)
    n_test = cfg.get("n_test")
    tt = test.iloc[:n_test] if n_test else test
    suffix = f"_{n_test}" if n_test else ""
    test_X = {
        lg: embed.cached(model, f"flores_devtest_{lg}{suffix}", tt[lg].tolist(), scale)
        for lg in langs
    }
    Ya = {
        lg: project(
            rel, embed.cached(ANCHOR, f"flores_devtest_{lg}{suffix}", tt[lg].tolist(), scale)
        )
        for lg in langs
    }
    return langs, test_X, Ya


def _certify(scale: str, model: str, cfg: dict, ad, prev_name: str, prev_ad, commands: bool):
    """Certificate, retrieval and (for command sets) held-out command agreement of ``ad`` and of
    the set it is compared with, on this machine's embeddings of the same test texts."""
    from sphota.certify import evaluate, grade

    langs, test_X, Ya = _test_sets(scale, model, cfg)
    m = evaluate({lg: ad(test_X[lg]) for lg in langs}, Ya)
    cert = {k: m[k] for k in CERT_KEYS}
    cert["grade"] = grade(m)["grade"]
    r10 = {"new": recall_at_10(ad(test_X["en"]), Ya["en"])}
    prev_cert = None
    if prev_ad is not None:
        r10["prev"] = recall_at_10(prev_ad(test_X["en"]), Ya["en"])
        if commands:  # compare on identical embeddings (GPU fp16 vs CPU fp32 differ slightly)
            pm = evaluate({lg: prev_ad(test_X[lg]) for lg in langs}, Ya)
            prev_cert = {k: pm[k] for k in CERT_KEYS}
            prev_cert["grade"] = grade(pm)["grade"]
    cert["retrieval_r10_en"] = round(r10["new"], 4)
    cmd = (
        command_eval(scale, model, cfg["multilingual"], {prev_name: prev_ad, "new": ad})
        if commands and prev_ad is not None
        else None
    )
    return cert, prev_cert, r10, cmd


def run(scale: str, models: list[str], set_name: str = "v1.1", certify_only: bool = False) -> dict:
    """Train (or, with ``certify_only``, re-certify the shipped files of) an adapter set."""
    from sphota.adapters import get_adapter, load_adapter
    from sphota.release import load_release, sha256_file
    from sphota.space import project

    S = SETS[set_name]
    rel = load_release()
    commands = S["cmd_per_lang"] is not False
    prev = S["prev"]
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
        fname = f"{cfg['short']}.npz"
        if certify_only:
            entry = registry["adapters"][model]
            ad = load_adapter(out_dir / entry["file"], sha256=entry["sha256"])
            ts = entry["temperature_scale"]
        else:
            texts = training_texts(scale, S["n_train"] or cfg["n_train"])
            Y = anchor_canonical(scale, texts)
            X = embed.cached(model, f"adapter-train-{len(texts)}", texts, scale)
            data = ["wikipedia-20231101 (12 langs)", "flores200-dev"]
            if commands:  # short commands (MASSIVE train, CC BY 4.0; test texts excluded)
                ct = command_texts(scale, cfg["multilingual"], per_lang=S["cmd_per_lang"])
                tag = f"cmd-train-{'ml' if cfg['multilingual'] else 'en'}-{len(ct)}"
                X = np.concatenate([X, embed.cached(model, tag, ct, scale)])
                Y = np.concatenate([Y, project(rel, embed.cached(ANCHOR, tag, ct, scale))])
                data.append(f"amazon-massive-train ({len(ct)} commands)")
            meta = {
                "model_license": cfg["license"],
                "multilingual": cfg["multilingual"],
                "prefix": embed.PREFIX.get(model, ""),
                "train": {"pairs": len(X), "data": data},
            }
            ts = calibrate_temperature_scale(rel, X, Y, fit=lambda a, b: fit_mlp2("tmp", a, b))
            ad = fit_mlp2(model, X, Y, meta={**meta, "temperature_scale": ts}, set_name=set_name)
            p = ad.save(out_dir / fname)
            ad = load_adapter(p)  # certify exactly what ships (float16 weights, numpy forward)
        try:
            prev_ad = get_adapter(model, prev)
        except Exception:  # models without an adapter in the previous set (D-046)
            prev_ad = None
        cert, prev_cert, r10, cmd = _certify(scale, model, cfg, ad, prev, prev_ad, commands)
        registry["adapters"][model] = {
            "short": cfg["short"],
            "file": fname,
            "sha256": sha256_file(out_dir / fname),
            "kind": "mlp2",
            "model_license": cfg["license"],
            "multilingual": cfg["multilingual"],
            "prefix": ad.prefix,
            **({"onnx": cfg["onnx"]} if "onnx" in cfg else {}),
            "temperature_scale": ts,
            "certificate": cert,
        }
        reg_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        if prev_cert is None:  # v1.1: the stored certificate of the previous set
            prev_dir = rel.root / "adapters" if prev == "v1" else OUT.parent / prev
            prev_reg = json.loads((prev_dir / "registry.json").read_text(encoding="utf-8"))
            prev_cert = prev_reg["adapters"].get(model, {}).get("certificate")
        results[model] = {
            set_name: cert,
            prev: prev_cert,
            "retrieval_r10_en": {prev: r10.get("prev"), set_name: r10["new"]},
            "seconds": round(time.time() - t0),
            "embeddings": "certify-only (this machine)" if certify_only else "training machine",
        }
        if cmd is not None:
            results[model]["commands"] = {prev: cmd[prev], set_name: cmd["new"]}
        log(
            f"{model}: L1 en "
            f"{(prev_cert['agreement_same_text_en'][0] if prev_cert else float('nan')):.3f}→{cert['agreement_same_text_en'][0]:.3f} "
            f"all {cert['agreement_same_text'][0]:.3f} r@10 {r10['new']:.3f} {cert['grade']}"
            + (
                f" commands l1_en {cmd[prev]['l1_en']:.3f}→{cmd['new']['l1_en']:.3f}"
                if cmd is not None
                else ""
            )
        )
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", choices=["tiny", "small", "medium", "full"], default="small")
    # the five registry models (bge-m3 has no certified v1 adapter to compare with)
    ap.add_argument("--models", nargs="*", default=[m for m in MODELS if m != "BAAI/bge-m3"])
    ap.add_argument("--set", choices=list(SETS), default="v1.1")
    ap.add_argument(
        "--certify-only",
        action="store_true",
        help="re-certify the shipped files of --set on this machine (no training)",
    )
    a = ap.parse_args()
    res = run(a.scale, a.models, a.set, a.certify_only)
    out = ROOT / "bench" / "results" / f"adapters-{a.set}-{a.scale}.json"
    if out.exists():  # merge: keep models trained in earlier runs
        res = {**json.loads(out.read_text(encoding="utf-8")), **res}
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"wrote {out}")
