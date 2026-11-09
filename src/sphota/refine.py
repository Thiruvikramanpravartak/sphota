# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-file refinement: groups fitted to one file, attached to the stable codes (D-033).

Sphota's fixed themes (the first code block) are the same for everyone, which makes them stable
across files and months but too coarse for narrow domains and too fine for scattered short
messages. Refinement groups *one file's* canonical vectors and attaches every group to a stable
parent code; each message keeps its own code unchanged.

Distances are measured in **translation units**: the canonical space (meaning-LDA, paper §3.2) is
scaled so that a sentence and its translation lie ``NOISE_UNIT`` apart (median over FLORES-200 dev
translation pairs, 12 languages); unrelated sentences lie about 2 units apart. Thresholds are
therefore comparable across datasets and were fixed once on development data.

Methods (pre-registered in D-033):

* ``"bisect"`` (M2): top-down bisecting 2-means; a group is split while its RMS radius exceeds
  ``tau`` translation units.
* ``"split_merge"`` (M3): start from the fixed themes, merge themes whose centres are within
  ``tau`` units, then split groups wider than ``tau`` as in ``bisect``.
* M1 (k-means with a silhouette-chosen k) lives in the research harness only.

Pure NumPy, deterministic and independent of row order (rows are processed in a canonical order).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NOISE_UNIT = 22.9  # median canonical distance between FLORES-200 dev translation pairs
DEFAULT_TAU = 1.0  # placeholder until the dev sweep picks it (D-033); see refine-study results


@dataclass
class Refined:
    """Result of :func:`refine`: one group id per row, plus per-group facts."""

    labels: np.ndarray  # (n,) group id per row, 0..G-1, groups ordered by size (largest first)
    sizes: list[int] = field(default_factory=list)
    representative: list[int] = field(default_factory=list)  # row nearest each group's centre
    parent: list[int] = field(default_factory=list)  # most common first-level cell per group

    @property
    def n_groups(self) -> int:
        return len(self.sizes)


def _order(Y: np.ndarray) -> np.ndarray:
    """A canonical row order (lexicographic on rounded coordinates): makes results shuffle-proof."""
    R = np.round(Y[:, : min(8, Y.shape[1])].astype(np.float64), 4)
    return np.lexsort(R.T[::-1])


def script_of(text: str) -> str:
    """Dominant writing system of a text (first word of the Unicode names of its letters)."""
    import unicodedata

    counts: dict[str, int] = {}
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            key = "CJK" if name.startswith(("CJK", "HIRAGANA", "KATAKANA")) else name.split(" ")[0]
            counts[key] = counts.get(key, 0) + 1
    return max(counts, key=lambda k: counts[k]) if counts else "NONE"


def centre_by_group(Y: np.ndarray, groups: np.ndarray, min_rows: int = 30) -> np.ndarray:
    """Remove each group's mean offset (e.g. per writing system), keeping the overall mean.

    Only groups with at least ``min_rows`` rows are shifted, and only when the file has two or
    more such groups; single-script files are returned unchanged.
    """
    groups = np.asarray(groups)
    big = [g for g in np.unique(groups) if (groups == g).sum() >= min_rows]
    if len(big) < 2:
        return Y
    out = Y.copy()
    mu = Y.mean(0)
    for g in big:
        m = groups == g
        out[m] += mu - Y[m].mean(0)
    return out


def _two_means(X: np.ndarray, iters: int = 12) -> np.ndarray:
    """Deterministic 2-means: seeds are the point farthest from the centre and the point farthest
    from that one. Returns a boolean mask for the second group."""
    c = X.mean(0)
    a = X[np.argmax(((X - c) ** 2).sum(1))]
    b = X[np.argmax(((X - a) ** 2).sum(1))]
    mask = ((X - b) ** 2).sum(1) < ((X - a) ** 2).sum(1)
    for _ in range(iters):
        if mask.all() or not mask.any():
            break
        a, b = X[~mask].mean(0), X[mask].mean(0)
        new = ((X - b) ** 2).sum(1) < ((X - a) ** 2).sum(1)
        if (new == mask).all():
            break
        mask = new
    return mask


def _radius(X: np.ndarray) -> float:
    return float(np.sqrt(((X - X.mean(0)) ** 2).sum(1).mean())) if len(X) > 1 else 0.0


def _bisect_groups(
    Y: np.ndarray,
    groups: list[np.ndarray],
    limit: float,
    min_size: int,
    guard: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Split groups (lists of row indices) while their RMS radius exceeds ``limit``.

    ``guard``: optional per-row script ids; a split whose halves are each >= 80% a different script
    is refused (it would separate languages rather than meanings).
    """
    done: list[np.ndarray] = []
    todo = [g for g in groups]
    while todo:
        g = todo.pop()
        if len(g) < 2 * min_size or _radius(Y[g]) <= limit:
            done.append(g)
            continue
        m = _two_means(Y[g])
        a, b = g[~m], g[m]
        lang_split = guard is not None and _language_split(guard, a, b)
        if len(a) < min_size or len(b) < min_size or lang_split:
            done.append(g)
            continue
        todo += [a, b]
    return done


def _language_split(scripts: np.ndarray, a: np.ndarray, b: np.ndarray, share: float = 0.8) -> bool:
    def dominant(ix: np.ndarray) -> tuple[object, float]:
        vals, counts = np.unique(scripts[ix], return_counts=True)
        i = int(np.argmax(counts))
        return vals[i], counts[i] / len(ix)

    sa, pa = dominant(a)
    sb, pb = dominant(b)
    return bool(sa != sb and pa >= share and pb >= share)


def _merge_close(Y: np.ndarray, groups: list[np.ndarray], limit: float) -> list[np.ndarray]:
    """Merge groups whose centres are within ``limit`` (single linkage on centres, union-find)."""
    if len(groups) < 2:
        return groups
    C = np.stack([Y[g].mean(0) for g in groups])
    parent = list(range(len(groups)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    cn = (C**2).sum(1)
    D2 = cn[:, None] + cn[None] - 2 * C @ C.T
    close = D2 <= limit**2  # noqa: SIM300
    ii, jj = np.where(np.triu(close, 1))
    for i, j in zip(ii.tolist(), jj.tolist()):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    merged: dict[int, list[np.ndarray]] = {}
    for i, g in enumerate(groups):
        merged.setdefault(find(i), []).append(g)
    return [np.sort(np.concatenate(v)) for v in merged.values()]


def silhouette(Y: np.ndarray, labels: np.ndarray, sample: int = 1000, seed: int = 0) -> float:
    """Mean silhouette of ``labels`` on a deterministic sample of rows (NumPy, Euclidean)."""
    n = len(Y)
    idx = np.sort(np.random.default_rng(seed).permutation(n)[: min(sample, n)])
    X, lab = Y[idx], np.asarray(labels)[idx]
    uniq = np.unique(lab)
    if len(uniq) < 2 or len(uniq) >= len(idx):
        return -1.0
    xn = (X**2).sum(1)
    D = np.sqrt(np.maximum(xn[:, None] + xn[None] - 2 * X @ X.T, 0))
    M = np.stack([lab == u for u in uniq], 1).astype(np.float64)  # (m, G)
    cnt = M.sum(0)
    S = D @ M  # sum of distances to each group
    own = (M * S).sum(1)
    own_cnt = (M * cnt).sum(1) - 1
    a = np.where(own_cnt > 0, own / np.maximum(own_cnt, 1), 0.0)
    mean_other = np.where(M > 0, np.inf, S / np.maximum(cnt, 1))
    b = mean_other.min(1)
    s = np.where(own_cnt > 0, (b - a) / np.maximum(np.maximum(a, b), 1e-12), 0.0)
    return float(s.mean())


AUTO_TAUS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def refine(
    Y: np.ndarray,
    method: str = "bisect",
    tau: float | None = None,
    l1: np.ndarray | None = None,
    min_size: int = 2,
    unit: float = NOISE_UNIT,
    scripts: np.ndarray | None = None,
    language: str = "none",
) -> Refined:
    """Group one file's canonical vectors ``Y`` (n, D).

    ``l1``: the rows' first-level cell indices (needed for ``split_merge`` and to name parents).
    ``tau``: threshold in translation units (default :data:`DEFAULT_TAU`).
    ``scripts``: per-row writing system (see :func:`script_of`), used by ``language``:
    ``"centre"`` removes each script's mean offset before grouping, ``"guard"`` refuses splits
    that separate scripts, ``"none"`` ignores scripts.
    """
    Y = np.asarray(Y, dtype=np.float64)
    if language not in ("none", "centre", "guard"):
        raise ValueError("language must be 'none', 'centre' or 'guard'")
    if language != "none" and scripts is None:
        raise ValueError(f"language={language!r} needs per-row scripts (scripts=...)")
    if language == "centre":
        Y = centre_by_group(Y, np.asarray(scripts))
    n = len(Y)
    if n == 0:
        return Refined(labels=np.zeros(0, dtype=np.int64))
    tau = DEFAULT_TAU if tau is None else float(tau)
    limit = tau * unit
    order = _order(Y)
    Yo = Y[order]
    l1o = None if l1 is None else np.asarray(l1)[order]
    guard = np.asarray(scripts)[order] if language == "guard" else None
    if method == "auto":
        # split_merge at the threshold whose grouping has the best silhouette (label-free)
        best, best_s = None, -2.0
        for t in AUTO_TAUS:
            r = refine(
                Y,
                "split_merge",
                tau=t,
                l1=l1,
                min_size=min_size,
                unit=unit,
                scripts=scripts,
                language="guard" if language == "guard" else "none",
            )
            s = silhouette(Y, r.labels)
            if s > best_s:
                best, best_s = r, s
        assert best is not None
        return best
    if method == "bisect":
        groups = _bisect_groups(Yo, [np.arange(n)], limit, min_size, guard)
    elif method == "split_merge":
        if l1o is None:
            raise ValueError("split_merge needs the rows' first-level cells (l1=...)")
        start = [np.where(l1o == c)[0] for c in np.unique(l1o)]
        groups = _bisect_groups(Yo, _merge_close(Yo, start, limit), limit, min_size, guard)
    else:
        raise ValueError(f"unknown method {method!r}; use 'bisect', 'split_merge' or 'auto'")
    # deterministic group order: by size, then by first row in canonical order
    groups.sort(key=lambda g: (-len(g), int(g.min())))
    lab_o = np.empty(n, dtype=np.int64)
    reps, parents = [], []
    for gi, g in enumerate(groups):
        lab_o[g] = gi
        c = Yo[g].mean(0)
        reps.append(int(order[g[np.argmin(((Yo[g] - c) ** 2).sum(1))]]))
        if l1o is not None:
            vals, counts = np.unique(l1o[g], return_counts=True)
            parents.append(int(vals[np.argmax(counts)]))
    labels = np.empty(n, dtype=np.int64)
    labels[order] = lab_o
    return Refined(
        labels=labels, sizes=[len(g) for g in groups], representative=reps, parent=parents
    )
