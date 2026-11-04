# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Canonical space and quantization engine (numpy only).

Given anchor-model vectors (or vectors already mapped into the canonical space by an
adapter), produce per-level code indices with calibrated confidences and alternates.

Confidence semantics (spec section 5): ``level_confidence[l]`` estimates the probability
that another expression of the same meaning (a translation, a paraphrase, or the same text
through another certified model) gets the same level-``l`` symbol *given the same prefix*.
``confidence[l]`` is the cumulative product: the probability that the whole prefix up to
level ``l`` matches. ``min_confidence`` truncates where the cumulative confidence drops below
the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sphota.release import Release

TOP_M = 16  # candidates considered per level for probabilities / stable mode
N_NOISE = 32  # noise samples for stable mode (fixed seed => deterministic)
ALT_MIN_P = 0.02


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def project(rel: Release, X: np.ndarray) -> np.ndarray:
    """Anchor-model vectors (n, d) -> canonical vectors (n, D)."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float32))
    if X.shape[1] != rel.W.shape[0]:
        raise ValueError(
            f"vector has {X.shape[1]} dims but the v1 anchor ({rel.anchor['model']}) has "
            f"{rel.W.shape[0]}. For other models use from_vector(vec, model=...) with an adapter.")
    return ((l2n(X) - rel.mu) @ rel.W).astype(np.float32)


@dataclass
class Quantized:
    indices: np.ndarray  # (n, L) int64
    level_conf: np.ndarray  # (n, L) float
    cand: list[np.ndarray]  # per level: (n, M) candidate indices (sorted by probability)
    cand_p: list[np.ndarray]  # per level: (n, M) probabilities

    @property
    def confidence(self) -> np.ndarray:
        return np.cumprod(self.level_conf, axis=1)


_NOISE_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _noise(dim: int, n: int = N_NOISE) -> np.ndarray:
    key = (dim, n)
    if key not in _NOISE_CACHE:
        rng = np.random.default_rng(20261104)
        _NOISE_CACHE[key] = rng.standard_normal((n, dim)).astype(np.float32)
    return _NOISE_CACHE[key]


def _topm(R: np.ndarray, C: np.ndarray, cn: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    d2 = cn[None, :] - 2.0 * (R.astype(np.float64) @ C.T.astype(np.float64))
    d2 += (R.astype(np.float64) ** 2).sum(1, keepdims=True)
    m = min(m, C.shape[0])
    part = np.argpartition(d2, m - 1, axis=1)[:, :m] if m < C.shape[0] else np.tile(
        np.arange(C.shape[0]), (len(R), 1))
    dd = np.take_along_axis(d2, part, 1)
    order = np.argsort(dd, axis=1, kind="stable")
    return np.take_along_axis(part, order, 1), np.take_along_axis(dd, order, 1)


def quantize(
    rel: Release,
    Y: np.ndarray,
    levels: int | None = None,
    stable: bool = False,
    chunk: int = 2048,
) -> Quantized:
    """Quantize canonical vectors ``Y`` (n, D)."""
    Y = np.atleast_2d(np.asarray(Y, dtype=np.float32))
    L = levels or len(rel.codebooks)
    n = len(Y)
    idx = np.zeros((n, L), np.int64)
    conf = np.zeros((n, L), np.float64)
    cands: list[np.ndarray] = [np.zeros((n, 0), np.int64)] * L
    cps: list[np.ndarray] = [np.zeros((n, 0))] * L
    for s in range(0, n, chunk):
        R = Y[s : s + chunk].copy()
        for lvl in range(L):
            C = rel.codebooks[lvl]
            cand, d2 = _topm(R, C, rel.cnorm(lvl), TOP_M)
            T = float(rel.temperatures[lvl])
            logits = -(d2 - d2[:, :1]) / T
            p = np.exp(logits)
            p /= p.sum(1, keepdims=True)
            choice = np.zeros(len(R), np.int64)  # position within candidates
            if stable and cand.shape[1] > 1:
                choice = _stable_choice(R, C, cand, rel.noise_scale)
                # re-rank so the chosen candidate comes first in alternates
            chosen = np.take_along_axis(cand, choice[:, None], 1)[:, 0]
            pc = np.take_along_axis(p, choice[:, None], 1)[:, 0]
            idx[s : s + chunk, lvl] = chosen
            conf[s : s + chunk, lvl] = pc
            order = np.argsort(-p, axis=1, kind="stable")
            if s == 0:
                cands[lvl] = np.zeros((n, cand.shape[1]), np.int64)
                cps[lvl] = np.zeros((n, cand.shape[1]))
            cands[lvl][s : s + chunk] = np.take_along_axis(cand, order, 1)
            cps[lvl][s : s + chunk] = np.take_along_axis(p, order, 1)
            R -= C[chosen]
    return Quantized(idx, conf, cands, cps)


def _stable_choice(R: np.ndarray, C: np.ndarray, cand: np.ndarray, sigma: float) -> np.ndarray:
    """Majority vote of nearest-candidate under fixed Gaussian perturbations (deterministic).

    Points deep inside a cell are unaffected; near-boundary points move to the cell that
    holds most of their noise neighbourhood, which is where re-encodings of the same meaning
    are most likely to land.
    """
    Z = _noise(R.shape[1]) * sigma  # (S, D)
    Cc = C[cand]  # (n, M, D)
    # distances of each noisy copy to each candidate: ||r + z - c||^2
    base = ((R[:, None, :] - Cc) ** 2).sum(-1)  # (n, M)
    cross = 2.0 * np.einsum("sd,nmd->nsm", Z, R[:, None, :] - Cc)  # (n, S, M)
    d = base[:, None, :] + cross  # + ||z||^2 is constant per sample
    votes = np.zeros(cand.shape, np.int64)
    winners = d.argmin(-1)  # (n, S)
    for j in range(cand.shape[1]):
        votes[:, j] = (winners == j).sum(1)
    # ties resolve toward the nearest centroid (position 0)
    return votes.argmax(1)


def topk_codes(q: Quantized, row: int, k: int, levels: int | None = None) -> list[tuple[tuple[int, ...], float]]:
    """Beam search over levels for the k most probable full codes of one row.

    Deeper levels' candidates are only exact for the chosen prefix; for alternative prefixes
    the deeper levels reuse the chosen path's top symbol (an approximation documented in the
    spec, section 6.3).
    """
    L = levels or q.indices.shape[1]
    beams: list[tuple[tuple[int, ...], float]] = [((), 1.0)]
    for lvl in range(L):
        new = []
        for prefix, pr in beams:
            on_path = prefix == tuple(int(v) for v in q.indices[row, :lvl])
            if on_path:
                opts = [(int(c), float(p)) for c, p in zip(q.cand[lvl][row], q.cand_p[lvl][row])
                        if p >= ALT_MIN_P]
            else:
                opts = [(int(q.indices[row, lvl]), 1.0)]
            for c, p in opts:
                new.append(((*prefix, c), pr * p))
        new.sort(key=lambda t: -t[1])
        beams = new[: max(k, 1) * 2]
    return beams[:k]
