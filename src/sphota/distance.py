# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Hierarchy-aware geometry on codes (numpy only; spec section 6).

A code (or prefix) denotes a point in the canonical space: the sum of its level centroids
(residual reconstruction). Shorter prefixes are coarser points. Distances between codes are
Euclidean distances between these points, so they need nothing but the shipped codebooks.
"""

from __future__ import annotations

import heapq

import numpy as np

from sphota.codes import Code, from_indices, parse
from sphota.release import Release, load_release


def reconstruct(code: Code | str, rel: Release | None = None) -> np.ndarray:
    rel = rel or load_release()
    c = parse(code)
    v = np.zeros(rel.dim, np.float64)
    for lvl, i in enumerate(c.levels):
        v += rel.codebooks[lvl][i]
    return v


def distance(a: Code | str, b: Code | str, rel: Release | None = None) -> float:
    """Euclidean distance between code points, in translation-noise units."""
    rel = rel or load_release()
    return float(np.linalg.norm(reconstruct(a, rel) - reconstruct(b, rel)))


def similarity(a: Code | str, b: Code | str, rel: Release | None = None) -> float:
    """Similarity in [0, 1]: cosine of the code points (the space is centered), clipped at 0.

    Two identical codes give 1.0; a code and its own prefix give a high but lower value.
    """
    rel = rel or load_release()
    va, vb = reconstruct(a, rel), reconstruct(b, rel)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(max(0.0, min(1.0, va @ vb / (na * nb))))


def _l1_table(rel: Release) -> tuple[np.ndarray, np.ndarray]:
    if rel.l1_neighbors is not None and rel.l1_neighbor_dist is not None:
        return rel.l1_neighbors, rel.l1_neighbor_dist
    C = rel.codebooks[0].astype(np.float64)
    d2 = (C**2).sum(1)[:, None] - 2 * C @ C.T + (C**2).sum(1)[None, :]
    np.fill_diagonal(d2, np.inf)
    nb = np.argsort(d2, 1)[:, :32]
    return nb, np.sqrt(np.maximum(np.take_along_axis(d2, nb, 1), 0))


def neighbors(code: Code | str, k: int = 10, rel: Release | None = None) -> list[tuple[Code, float]]:
    """Nearest codes at the same depth as ``code`` (sorted by distance)."""
    rel = rel or load_release()
    c = parse(code)
    depth = c.depth
    if depth == 1:
        nb, dist = _l1_table(rel)
        return [(from_indices([int(j)]), float(d)) for j, d in zip(nb[c.levels[0]][:k], dist[c.levels[0]][:k])]
    # same parent prefix, vary the last level; plus neighbouring L1 cells' best matches
    v = reconstruct(c, rel)
    parent = c.levels[:-1]
    base = reconstruct(from_indices(list(parent)), rel)
    C = rel.codebooks[depth - 1]
    cand = base[None, :] + C
    d = np.linalg.norm(cand - v[None, :], axis=1)
    out = [(from_indices([*parent, int(j)]), float(d[j])) for j in np.argsort(d) if j != c.levels[-1]]
    return out[:k]


def path(a: Code | str, b: Code | str, rel: Release | None = None, max_hops: int = 12) -> list[Code]:
    """A semantic route from ``a`` to ``b`` through neighbouring L1 regions (Dijkstra on the
    shipped L1 k-NN graph), ending at ``b`` itself."""
    rel = rel or load_release()
    ca, cb = parse(a), parse(b)
    s, t = ca.levels[0], cb.levels[0]
    nb, dist = _l1_table(rel)
    best = {s: 0.0}
    prev: dict[int, int] = {}
    hops = {s: 0}
    heap = [(0.0, s)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == t:
            break
        if d > best.get(u, np.inf) or hops[u] >= max_hops:
            continue
        for v, w in zip(nb[u][:12], dist[u][:12]):
            v = int(v)
            nd = d + float(w) ** 2  # squared edge weights favour many small steps
            if nd < best.get(v, np.inf):
                best[v] = nd
                prev[v] = u
                hops[v] = hops[u] + 1
                heapq.heappush(heap, (nd, v))
    route: list[Code] = [cb]
    if t in prev or t == s:
        u = t
        while u != s:
            u = prev[u]
            route.append(from_indices([u]) if u != s else ca)
        if s == t:
            route = [cb] if ca == cb else [ca, cb]
            return route
        return list(reversed(route))
    # disconnected within hop budget: direct jump
    return [ca, cb]
