# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Fitting the canonical space and the quantizers (offline, numpy + scikit-learn).

Canonical space
---------------
``y = (normalize(x) - mu) @ W`` where ``W`` (d_model x D) is the product of

1. PCA to ``P`` dims (removes noise directions, makes step 2 well-conditioned), and
2. *meaning-LDA*: the generalized eigenvectors of ``(S_total, S_within + reg)``, where
   ``S_within`` is the scatter of translations of the same sentence around their mean
   (cross-lingual "noise") and ``S_total`` is the corpus scatter ("signal"). Directions are
   scaled so that the within-meaning noise has unit variance. Euclidean distance in this
   space is therefore measured in "translation-noise units", and k-means boundaries are
   placed where languages agree.

Quantizers
----------
* ``rkm``  residual k-means (global residual codebooks per level),
* ``tree`` hierarchical k-means (each child codebook conditioned on its parent),
* ``rqema`` residual quantization learned online with EMA updates and dead-code revival
  (the RQ-VAE recipe with identity encoder/decoder, which is what RQ-VAE reduces to on a
  frozen space).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


# ---------------------------------------------------------------- canonical space


@dataclass
class Space:
    mu: np.ndarray  # (d,)
    W: np.ndarray  # (d, D)
    eigvals: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((l2n(x) - self.mu) @ self.W).astype(np.float32)


def within_scatter(groups: list[np.ndarray]) -> tuple[np.ndarray, int]:
    d = groups[0].shape[1]
    S = np.zeros((d, d))
    n = 0
    for g in groups:
        if len(g) < 2:
            continue
        c = g - g.mean(0, keepdims=True)
        S += c.T @ c
        n += len(g) - 1
    return S / max(n, 1), n


def fit_space(
    corpus: np.ndarray,
    groups: list[np.ndarray] | None,
    D: int,
    P: int | None = None,
    reg: float = 0.1,
    model_residuals: np.ndarray | None = None,
    model_weight: float = 1.0,
) -> Space:
    """Fit the canonical projection. ``groups``: list of (n_i, d) arrays of translations.

    ``model_residuals`` (n, d): differences between other models' (linearly mapped) vectors
    and the anchor's vectors for the same texts. Their scatter is added to the within-meaning
    noise with relative weight ``model_weight`` (trace-normalized), so the space down-weights
    directions where *models* disagree as well as directions where *languages* disagree.

    With ``groups=None`` this is plain PCA-whitening (the "no alignment" baseline).
    """
    X = l2n(corpus.astype(np.float64))
    mu = X.mean(0)
    Xc = X - mu
    P = min(P or 2 * D, X.shape[1])
    # PCA
    cov = Xc.T @ Xc / len(Xc)
    ev, V = np.linalg.eigh(cov)
    idx = np.argsort(ev)[::-1][:P]
    Vp = V[:, idx]
    if groups is None:
        W = Vp[:, :D] / np.sqrt(ev[idx][:D])[None, :]
        return Space(mu.astype(np.float32), W.astype(np.float32), ev[idx][:D])
    St = Vp.T @ cov @ Vp
    Sw, _ = within_scatter([(l2n(g.astype(np.float64)) - mu) @ Vp for g in groups])
    if model_residuals is not None and model_weight > 0:
        Rp = model_residuals.astype(np.float64) @ Vp
        Sm = Rp.T @ Rp / len(Rp)
        Sw = Sw + model_weight * Sm * np.trace(Sw) / np.trace(Sm)
    # shrinkage toward isotropic noise keeps the solution stable with few meanings
    Sw = (1 - reg) * Sw + reg * np.trace(Sw) / P * np.eye(P)
    # generalized eigenproblem St v = lambda Sw v  via Cholesky whitening of Sw
    L = np.linalg.cholesky(Sw)
    Li = np.linalg.inv(L)
    M = Li @ St @ Li.T
    lam, U = np.linalg.eigh((M + M.T) / 2)
    order = np.argsort(lam)[::-1][:D]
    Vg = Li.T @ U[:, order]  # columns satisfy v^T Sw v = 1
    W = Vp @ Vg
    return Space(mu.astype(np.float32), W.astype(np.float32), lam[order])


# ---------------------------------------------------------------- k-means helpers


def kmeans(X: np.ndarray, k: int, seed: int = 0, iters: int = 30, sample: int | None = 200_000):
    from sklearn.cluster import KMeans

    k = min(k, len(X))
    rng = np.random.default_rng(seed)
    Xs = X if sample is None or len(X) <= sample else X[rng.choice(len(X), sample, replace=False)]
    km = KMeans(n_clusters=k, n_init=1, max_iter=iters, random_state=seed, init="k-means++")
    km.fit(Xs.astype(np.float32))
    return km.cluster_centers_.astype(np.float32)


def assign(X: np.ndarray, C: np.ndarray, chunk: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Nearest centroid and squared distances to the 2 nearest (for margins)."""
    cn = (C * C).sum(1)
    idx = np.empty(len(X), np.int64)
    d2 = np.empty((len(X), 2), np.float32)
    for s in range(0, len(X), chunk):
        x = X[s : s + chunk]
        D = cn[None, :] - 2 * x @ C.T + (x * x).sum(1, keepdims=True)
        if D.shape[1] > 1:
            part = np.argpartition(D, 1, axis=1)[:, :2]
            dd = np.take_along_axis(D, part, 1)
            o = np.argsort(dd, 1)
            part = np.take_along_axis(part, o, 1)
            dd = np.take_along_axis(dd, o, 1)
        else:
            part = np.zeros((len(x), 2), np.int64)
            dd = np.repeat(D, 2, 1)
        idx[s : s + chunk] = part[:, 0]
        d2[s : s + chunk] = dd
    return idx, d2


# ---------------------------------------------------------------- quantizers


class Quantizer:
    name = "base"
    K: tuple[int, ...]

    def fit(self, X: np.ndarray) -> Quantizer:  # pragma: no cover - interface
        raise NotImplementedError

    def encode(self, X: np.ndarray) -> np.ndarray:  # (n, L) int
        raise NotImplementedError


class ResidualKMeans(Quantizer):
    name = "rkm"

    def __init__(self, K: tuple[int, ...], seed: int = 0, iters: int = 30):
        self.K, self.seed, self.iters = K, seed, iters
        self.codebooks: list[np.ndarray] = []

    def fit(self, X: np.ndarray) -> ResidualKMeans:
        R = X.astype(np.float32).copy()
        self.codebooks = []
        for lvl, k in enumerate(self.K):
            C = kmeans(R, k, seed=self.seed + lvl, iters=self.iters)
            idx, _ = assign(R, C)
            R -= C[idx]
            self.codebooks.append(C)
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        R = X.astype(np.float32).copy()
        out = np.empty((len(X), len(self.codebooks)), np.int64)
        for lvl, C in enumerate(self.codebooks):
            idx, _ = assign(R, C)
            out[:, lvl] = idx
            R -= C[idx]
        return out


class RQEMA(ResidualKMeans):
    """Residual quantizer trained online with EMA codebooks + dead-code revival."""

    name = "rqema"

    def __init__(
        self, K, seed=0, epochs=8, batch=1024, decay=0.95, revive_below=1e-3, init_random=True
    ):
        super().__init__(K, seed)
        self.epochs, self.batch, self.decay, self.revive_below = epochs, batch, decay, revive_below
        self.init_random = init_random

    def fit(self, X: np.ndarray) -> RQEMA:
        rng = np.random.default_rng(self.seed)
        R = X.astype(np.float32).copy()
        self.codebooks = []
        for k in self.K:
            k = min(k, len(R))
            # RQ-VAE style random init (from data points) -- the source of collapse
            C = R[rng.choice(len(R), k, replace=False)].copy()
            N = np.ones(k, np.float32)
            S = C.copy()
            for _ in range(self.epochs):
                perm = rng.permutation(len(R))
                for s in range(0, len(R), self.batch):
                    xb = R[perm[s : s + self.batch]]
                    idx, _ = assign(xb, C)
                    cnt = np.bincount(idx, minlength=k).astype(np.float32)
                    sm = np.zeros_like(C)
                    np.add.at(sm, idx, xb)
                    N = self.decay * N + (1 - self.decay) * cnt
                    S = self.decay * S + (1 - self.decay) * sm
                    C = S / np.maximum(N, 1e-6)[:, None]
                    # dead-code revival: re-seed rarely used codes from current batch
                    dead = np.where(N / N.sum() < self.revive_below / k)[0]
                    if len(dead):
                        C[dead] = xb[rng.choice(len(xb), len(dead))]
                        N[dead] = N.mean()
                        S[dead] = C[dead] * N[dead, None]
            idx, _ = assign(R, C)
            R -= C[idx]
            self.codebooks.append(C.astype(np.float32))
        return self


class TreeKMeans(Quantizer):
    """Hierarchical k-means: children conditioned on parent (codebook per node)."""

    name = "tree"

    def __init__(self, K: tuple[int, ...], seed: int = 0, min_per_child: int = 4):
        self.K, self.seed, self.min_per_child = K, seed, min_per_child
        self.root: np.ndarray | None = None
        self.children: list[dict[tuple[int, ...], np.ndarray]] = []

    def fit(self, X: np.ndarray) -> TreeKMeans:
        self.root = kmeans(X, self.K[0], seed=self.seed)
        idx, _ = assign(X, self.root)
        paths = idx[:, None]
        self.children = []
        for lvl, k in enumerate(self.K[1:], start=1):
            level: dict[tuple[int, ...], np.ndarray] = {}
            new = np.zeros(len(X), np.int64)
            keys = [tuple(p) for p in paths]
            groups: dict[tuple[int, ...], list[int]] = {}
            for i, key in enumerate(keys):
                groups.setdefault(key, []).append(i)
            for key, members in groups.items():
                m = np.array(members)
                kk = max(1, min(k, len(m) // self.min_per_child))
                C = (
                    kmeans(X[m], kk, seed=self.seed + lvl, iters=20)
                    if kk > 1
                    else X[m].mean(0, keepdims=True)
                )
                level[key] = C.astype(np.float32)
                new[m] = assign(X[m], C)[0]
            self.children.append(level)
            paths = np.concatenate([paths, new[:, None]], 1)
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        assert self.root is not None
        out = np.zeros((len(X), 1 + len(self.children)), np.int64)
        out[:, 0] = assign(X, self.root)[0]
        for lvl, level in enumerate(self.children, start=1):
            keys = [tuple(r) for r in out[:, :lvl]]
            for i, key in enumerate(keys):
                C = level.get(key)
                if C is None:
                    out[i, lvl] = 0
                else:
                    out[i, lvl] = int(np.argmin(((C - X[i]) ** 2).sum(1)))
        return out


QUANTIZERS = {"rkm": ResidualKMeans, "rqema": RQEMA, "tree": TreeKMeans}


# ---------------------------------------------------------------- metrics


def prefix_agree(a: np.ndarray, b: np.ndarray) -> list[float]:
    """Fraction of rows whose codes agree on the first l levels, for each l."""
    eq = a == b
    return [float(np.all(eq[:, : lvl + 1], 1).mean()) for lvl in range(a.shape[1])]


def utilization(codes: np.ndarray, K1: int) -> dict:
    c = np.bincount(codes[:, 0], minlength=K1)
    p = c[c > 0] / c.sum()
    ent = float(-(p * np.log(p)).sum())
    return {"l1_used_frac": float((c > 0).mean()), "l1_perplexity_frac": float(np.exp(ent) / K1)}


def nmi(a: np.ndarray, b: np.ndarray) -> float:
    from sklearn.metrics import normalized_mutual_info_score

    return float(normalized_mutual_info_score(a, b))


def retrieval_top1(Q: np.ndarray, D: np.ndarray) -> float:
    """Q[i] should retrieve D[i] (cosine)."""
    s = l2n(Q) @ l2n(D).T
    return float((s.argmax(1) == np.arange(len(Q))).mean())


def ridge_map(X: np.ndarray, Y: np.ndarray, alpha: float = 1.0):
    """Ridge regression from L2-normalized X to Y; returns a callable."""
    Xn = l2n(X.astype(np.float64))
    mx, my = Xn.mean(0), Y.mean(0)
    G = (Xn - mx).T @ (Xn - mx) + alpha * len(Xn) / 1000 * np.eye(Xn.shape[1])
    A = np.linalg.solve(G, (Xn - mx).T @ (Y - my))
    return lambda Z: (l2n(Z.astype(np.float64)) - mx) @ A + my


def cross_model_residuals(anchor: np.ndarray, others: list[np.ndarray]) -> np.ndarray:
    """Cross-fitted residuals (2-fold) of linear maps from each model into the anchor's raw
    (normalized) space; cross-fitting avoids under-estimating the residual scatter."""
    Ya = l2n(anchor.astype(np.float64))
    n = len(Ya)
    half = np.arange(n) < n // 2
    out = []
    for X in others:
        f1, f2 = ridge_map(X[half], Ya[half]), ridge_map(X[~half], Ya[~half])
        out.append(np.concatenate([f2(X[half]) - Ya[half], f1(X[~half]) - Ya[~half]]))
    return np.concatenate(out)
