# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Unsupervised adapter experiment (vec2vec-style, no paired data). Milestone M4.

    python -m sphota.build.vec2vec --scale small --model sentence-transformers/all-MiniLM-L6-v2

Simplified from Jha et al. 2025 (arXiv:2505.12540): two input adapters into a shared latent,
two output heads, trained with
* adversarial losses on translated vectors (both directions, non-saturating GAN),
* reconstruction (b → latent → b) and cycle consistency (b → canonical → b),
* vector-space preservation (VSP): pairwise cosine structure of a batch is kept under
  translation.

The model side and the anchor side see **disjoint** text sets (first and second half of the
training texts), so no pair is ever observed. Evaluation: FLORES devtest English, against the
anchor's canonical vectors: top-1 retrieval among 1012 candidates, mean cosine, and per-level
code agreement. A supervised ridge adapter on the same budget is the upper reference.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from sphota.build import embed
from sphota.build.adapters import training_texts
from sphota.build.common import ROOT, log
from sphota.build.release import load


def mlp(i: int, o: int, h: int = 512):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(i, h), nn.SiLU(), nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, o)
    )


def run(scale: str, model: str, steps: int, seed: int = 0) -> dict:
    import torch
    import torch.nn.functional as F

    from sphota.certify import fit_adapter
    from sphota.release import load_release
    from sphota.space import project, quantize

    torch.manual_seed(seed)
    rel = load_release()
    langs, _dev, test, _w, _t = load(scale)
    texts = training_texts(scale, 40000)
    XB = embed.cached(model, f"adapter-train-{len(texts)}", texts, scale)
    XA = embed.cached("intfloat/multilingual-e5-small", f"adapter-train-{len(texts)}", texts, scale)
    YA = project(rel, XA)
    n = len(texts)
    b_idx, a_idx = np.arange(0, n // 2), np.arange(n // 2, n)  # disjoint: unpaired
    sa = float(np.sqrt((YA**2).sum(1).mean()))
    B = torch.tensor(
        XB[b_idx] / np.linalg.norm(XB[b_idx], axis=1, keepdims=True), dtype=torch.float32
    )
    A = torch.tensor(YA[a_idx] / sa, dtype=torch.float32)
    dB, dA, dz = B.shape[1], A.shape[1], 256
    inB, inA, outB, outA = mlp(dB, dz), mlp(dA, dz), mlp(dz, dB), mlp(dz, dA)
    discA, discB = mlp(dA, 1, 256), mlp(dB, 1, 256)
    gen = (
        list(inB.parameters())
        + list(inA.parameters())
        + list(outB.parameters())
        + list(outA.parameters())
    )
    og = torch.optim.Adam(gen, lr=2e-4, betas=(0.5, 0.999))
    od = torch.optim.Adam(
        list(discA.parameters()) + list(discB.parameters()), lr=2e-4, betas=(0.5, 0.999)
    )
    bce = torch.nn.BCEWithLogitsLoss()

    def cosmat(x):
        x = F.normalize(x, dim=1)
        return x @ x.T

    t0 = time.time()
    for step in range(steps):
        b = B[torch.randint(len(B), (256,))]
        a = A[torch.randint(len(A), (256,))]
        b2a = outA(inB(b))
        a2b = outB(inA(a))
        # discriminators
        od.zero_grad()
        ld = (
            bce(discA(a), torch.ones(256, 1))
            + bce(discA(b2a.detach()), torch.zeros(256, 1))
            + bce(discB(b), torch.ones(256, 1))
            + bce(discB(a2b.detach()), torch.zeros(256, 1))
        )
        ld.backward()
        od.step()
        # generators
        og.zero_grad()
        adv = bce(discA(b2a), torch.ones(256, 1)) + bce(discB(a2b), torch.ones(256, 1))
        rec = F.mse_loss(outB(inB(b)), b) * dB + F.mse_loss(outA(inA(a)), a) * dA / 10
        cyc = F.mse_loss(outB(inA(b2a)), b) * dB + F.mse_loss(outA(inB(a2b)), a) * dA / 10
        vsp = F.mse_loss(cosmat(b2a), cosmat(b)) + F.mse_loss(cosmat(a2b), cosmat(a))
        loss = adv + 10 * rec + 10 * cyc + 20 * vsp
        loss.backward()
        og.step()
        if step % 500 == 0:
            log(
                f"step {step} adv {adv.item():.3f} rec {rec.item():.3f} cyc {cyc.item():.3f} vsp {vsp.item():.4f}"
            )
    # evaluation on FLORES devtest English
    TB = embed.cached(model, "flores_devtest_en", test["en"].tolist(), scale)
    TA = project(
        rel,
        embed.cached(
            "intfloat/multilingual-e5-small", "flores_devtest_en", test["en"].tolist(), scale
        ),
    )
    with torch.no_grad():
        pred = (
            outA(
                inB(
                    torch.tensor(
                        TB / np.linalg.norm(TB, axis=1, keepdims=True), dtype=torch.float32
                    )
                )
            ).numpy()
            * sa
        )

    def metrics(P):
        Pn = P / np.linalg.norm(P, axis=1, keepdims=True)
        Tn = TA / np.linalg.norm(TA, axis=1, keepdims=True)
        S = Pn @ Tn.T
        top1 = float((S.argmax(1) == np.arange(len(S))).mean())
        cos = float(np.diag(S).mean())
        qa, qb = quantize(rel, P).indices, quantize(rel, TA).indices
        agree = [
            round(float(np.all(qa[:, : i + 1] == qb[:, : i + 1], 1).mean()), 4) for i in range(4)
        ]
        return {"top1_retrieval": round(top1, 4), "mean_cosine": round(cos, 4), "agreement": agree}

    sup = fit_adapter(model, XB[b_idx], YA[b_idx])  # supervised reference, same 20k budget
    res = {
        "model": model,
        "steps": steps,
        "unpaired_texts_per_side": len(b_idx),
        "unsupervised": metrics(pred),
        "supervised_ridge_reference": metrics(sup(TB)),
        "chance_top1": round(1 / len(TB), 5),
        "seconds": round(time.time() - t0),
    }
    log(json.dumps(res))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="small")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--steps", type=int, default=6000)
    a = ap.parse_args()
    res = run(a.scale, a.model, a.steps)
    (ROOT / "bench" / "results" / f"vec2vec-{a.scale}.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
