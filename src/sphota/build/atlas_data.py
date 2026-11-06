# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Export data for the Meaning Atlas (atlas/public/data).

    python -m sphota.build.atlas_data

* ``map.json``: L1 cells (code, x, y, size, region, labels, examples, neighbours),
  populated L2 cells (code, x, y, labels), the 32 product regions (id, name, color, centre).
* ``regions.json``: region centroids + temperature, so the browser can name a sentence's region.
* ``space.bin``: float32 mu (384), W (384x256), codebooks L1..L4 (row-major), little endian.
* ``space.json``: shapes, offsets, temperatures, anchor info, permutation constants.

The 2D layout is t-SNE on L1 centroids (cosine). Each L1 cell is coloured by the region
(``sphota.regions``, R00..R31) of its centroid; a region's label sits at its densest cluster. L2 cells are placed around their parent along the first two
principal directions of the parent's L2 offsets.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sphota.build.common import ROOT, log
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
    "#ff8787",
    "#ffc078",
    "#8ce99a",
    "#66d9e8",
    "#91a7ff",
    "#e599f7",
    "#faa2c1",
    "#c0eb75",
]


def regions_layer(C1: np.ndarray, xy: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, list]:
    """Region id per L1 cell, and one map entry per region (label at its densest cluster)."""
    from sphota.regions import _load, region_of

    _, _, names = _load()
    reg = np.array([int(region_of(C1[i])[0][1:]) for i in range(len(C1))])
    out = []
    for k in range(len(names)):
        m = np.where(reg == k)[0]
        if not len(m):
            continue
        P, w = xy[m], counts[m] + 1.0
        d2 = ((P[:, None] - P[None]) ** 2).sum(-1)
        dens = (np.exp(-d2 / (2 * 0.06**2)) * w[None]).sum(1)
        cx, cy = P[int(np.argmax(dens))]
        core = d2[int(np.argmax(dens))] < 0.2**2
        spread = (
            float(np.sqrt(((P[core] - [cx, cy]) ** 2).sum(1).mean())) if core.sum() > 1 else 0.05
        )
        full = names[str(k)]
        out.append(
            {
                "k": k,
                "id": f"R{k:02d}",
                "x": round(float(cx), 4),
                "y": round(float(cy), 4),
                "color": PALETTE[k % len(PALETTE)],
                "name": full.split(" / ")[0],
                "full": full,
                "members": len(m),
                "spread": round(max(spread, 0.04), 4),
            }
        )
    return reg, out


def write_regions(out: Path) -> None:
    """Region centroids for in-browser region lookup (mirrors sphota.regions.region_of)."""
    from sphota.regions import _load, _meta

    C, T, names = _load()
    (out / "regions.json").write_text(
        json.dumps(
            {
                "temperature": T,
                "dim": int(C.shape[1]),
                "centroids": [round(float(v), 5) for v in C.ravel()],
                "labels": names,
                "labels_i18n": _meta().get("labels_i18n", {}),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def add_regions(out: Path) -> None:
    """Re-colour an existing map.json by region without re-running t-SNE (keeps the layout)."""
    rel = load_release()
    d = json.loads((out / "map.json").read_text(encoding="utf-8"))
    xy = np.array([[c["x"], c["y"]] for c in d["cells"]])
    counts = np.array([c["n"] for c in d["cells"]], dtype=np.float64)
    reg, regions = regions_layer(rel.codebooks[0].astype(np.float64), xy, counts)
    for c in d["cells"]:
        c.pop("c", None)
        c["r"] = int(reg[c["i"]])
    d.pop("continents", None)
    d["regions"] = regions
    (out / "map.json").write_text(
        json.dumps(d, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    write_regions(out)
    log(f"atlas regions: {len(regions)} regions → {out}")


def build(out: Path) -> None:
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
    counts = np.load(rel.root / "neighbors.npz")["l1_counts"]
    reg, regions = regions_layer(C1, xy, counts)
    examples = json.loads((rel.root / "examples.json").read_text(encoding="utf-8"))
    labels = rel.labels
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
                "r": int(reg[i]),
                "l": lab,
                "ex": examples.get(str(i), [])[:2],
                "nb": [int(j) for j in rel.l1_neighbors[i][:6]],
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
            {"space": rel.version, "cells": cells, "regions": regions, "l2": l2},
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
    write_regions(out)
    log(f"atlas data: {len(cells)} L1, {len(l2)} L2, {len(regions)} regions → {out}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regions-only", action="store_true", help="re-colour map.json, keep layout")
    a = ap.parse_args()
    out = ROOT / "atlas" / "public" / "data"
    add_regions(out) if a.regions_only else build(out)
