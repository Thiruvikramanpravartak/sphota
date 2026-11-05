# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Export data for the Meaning Atlas (atlas/public/data).

    python -m sphota.build.atlas_data

* ``map.json``: L1 cells (code, x, y, size, continent, labels, examples, neighbours),
  populated L2 cells (code, x, y, labels), continents (name, color, centre).
* ``space.bin``: float32 mu (384), W (384x256), codebooks L1..L4 (row-major), little endian.
* ``space.json``: shapes, offsets, temperatures, anchor info, permutation constants.

The 2D layout is t-SNE on L1 centroids (cosine), continents are agglomerative (Ward)
clusters of the centroids. L2 cells are placed around their parent along the first two
principal directions of the parent's L2 offsets.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sphota.build.common import DATA_DIR, ROOT, log
from sphota.codes import PERM_A, PERM_B, WIDTHS, from_indices
from sphota.release import load_release

PALETTE = [
    "#ff6b6b",
    "#ffa94d",
    "#ffd43b",
    "#a9e34b",
    "#69db7c",
    "#38d9a9",
    "#3bc9db",
    "#4dabf7",
    "#748ffc",
    "#9775fa",
    "#da77f2",
    "#f783ac",
    "#e8590c",
    "#f08c00",
    "#66a80f",
    "#0ca678",
    "#1098ad",
    "#1c7ed6",
    "#4263eb",
    "#7048e8",
    "#ae3ec9",
    "#d6336c",
    "#c2255c",
    "#5c940d",
]


def build(out: Path, n_continents: int = 24) -> None:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.manifold import TSNE

    rel = load_release()
    C1 = rel.codebooks[0].astype(np.float64)
    C1n = C1 / np.linalg.norm(C1, axis=1, keepdims=True)
    log("t-SNE on L1 centroids")
    xy = TSNE(
        n_components=2, metric="cosine", perplexity=30, init="pca", random_state=0, max_iter=1500
    ).fit_transform(C1n)
    xy -= xy.mean(0)
    xy /= np.abs(xy).max()
    cont = AgglomerativeClustering(n_clusters=n_continents, linkage="ward").fit_predict(C1n)
    counts = np.load(rel.root / "neighbors.npz")["l1_counts"]
    examples = json.loads((rel.root / "examples.json").read_text(encoding="utf-8"))
    labels = rel.labels
    cache = DATA_DIR / "small" / "build"
    cont_names = {}
    p = cache / "continent_names.json"
    if p.exists():
        cont_names = {int(k): v for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
    cells = []
    for i in range(rel.sizes[0]):
        lab = labels.get(str(i), {})
        cells.append(
            {
                "i": i,
                "code": str(from_indices([i])),
                "x": round(float(xy[i, 0]), 4),
                "y": round(float(xy[i, 1]), 4),
                "n": int(counts[i]),
                "c": int(cont[i]),
                "l": lab,
                "ex": examples.get(str(i), [])[:2],
                "nb": [int(j) for j in rel.l1_neighbors[i][:6]],
            }
        )
    # continents: centre = mean position of members (weighted), name from cache or top member labels
    conts = []
    for k in range(n_continents):
        m = np.where(cont == k)[0]
        w = counts[m] + 1
        cx, cy = (xy[m] * w[:, None]).sum(0) / w.sum()
        top = [labels.get(str(i), {}).get("en", "") for i in m[np.argsort(-counts[m])][:12]]
        conts.append(
            {
                "k": k,
                "x": round(float(cx), 4),
                "y": round(float(cy), 4),
                "color": PALETTE[k % len(PALETTE)],
                "name": cont_names.get(k, top[0] if top else f"Region {k}"),
                "members": len(m),
                "top": [t for t in top if t][:8],
            }
        )
    # L2: place populated, labelled L2 cells around their parent
    l2 = []
    spread = 0.012
    by_parent: dict[int, list[int]] = {}
    for key in labels:
        if "." in key:
            a, b = key.split(".")
            by_parent.setdefault(int(a), []).append(int(b))
    C2 = rel.codebooks[1].astype(np.float64)
    for a, kids in by_parent.items():
        V = C2[kids]
        if len(kids) >= 2:
            Vc = V - V.mean(0)
            _, _, vt = np.linalg.svd(Vc, full_matrices=False)
            P = Vc @ vt[:2].T
            P /= np.abs(P).max() + 1e-9
        else:
            P = np.zeros((1, 2))
        for b, (px, py) in zip(kids, P):
            l2.append(
                {
                    "code": str(from_indices([a, b])),
                    "p": a,
                    "x": round(float(xy[a, 0] + spread * px), 5),
                    "y": round(float(xy[a, 1] + spread * py), 5),
                    "l": labels[f"{a}.{b}"],
                }
            )
    out.mkdir(parents=True, exist_ok=True)
    (out / "map.json").write_text(
        json.dumps(
            {"space": rel.version, "cells": cells, "continents": conts, "l2": l2},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    # binary space for in-browser encoding
    parts = [rel.mu.astype("<f4"), rel.W.astype("<f4")] + [c.astype("<f4") for c in rel.codebooks]
    offsets, off = [], 0
    for arr in parts:
        offsets.append(off)
        off += arr.size * 4
    with open(out / "space.bin", "wb") as f:
        for arr in parts:
            f.write(np.ascontiguousarray(arr).tobytes())
    meta = {
        "space": rel.version,
        "anchor": rel.anchor,
        "browser_model": "Xenova/multilingual-e5-small",
        "dim_in": int(rel.W.shape[0]),
        "dim": rel.dim,
        "sizes": list(rel.sizes),
        "offsets": {"mu": offsets[0], "W": offsets[1], "codebooks": offsets[2:]},
        "temperatures": [float(t) for t in rel.temperatures],
        "top_m": 16,
        "widths": list(WIDTHS),
        "perm_a": list(PERM_A),
        "perm_b": list(PERM_B),
    }
    (out / "space.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"atlas data: {len(cells)} L1, {len(l2)} L2, {n_continents} continents → {out}")


if __name__ == "__main__":
    build(ROOT / "atlas" / "public" / "data")
