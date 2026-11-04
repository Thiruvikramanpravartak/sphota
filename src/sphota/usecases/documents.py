# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Document fingerprint comparison: similarity matrix + topic diffs for a set of documents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sphota.api import Fingerprint, FingerprintDiff, compare, diff, fingerprint


@dataclass
class DocumentComparison:
    names: list[str]
    fingerprints: list[Fingerprint]
    similarity: np.ndarray

    def most_similar(self, name: str, k: int = 3) -> list[tuple[str, float]]:
        i = self.names.index(name)
        order = [j for j in np.argsort(-self.similarity[i]) if j != i][:k]
        return [(self.names[j], float(self.similarity[i, j])) for j in order]

    def diff(self, a: str, b: str, threshold: float = 0.05) -> FingerprintDiff:
        fa = self.fingerprints[self.names.index(a)]
        fb = self.fingerprints[self.names.index(b)]
        return diff(fa, fb, level=fa.level, threshold=threshold)

    def table(self) -> str:
        w = max(len(n) for n in self.names)
        lines = [" " * w + "  " + "  ".join(f"{n[:6]:>6}" for n in self.names)]
        for n, row in zip(self.names, self.similarity):
            lines.append(f"{n:<{w}}  " + "  ".join(f"{v:6.2f}" for v in row))
        return "\n".join(lines)


def compare_documents(
    docs: Mapping[str, str] | Sequence[str | Path], level: int = 2
) -> DocumentComparison:
    """Fingerprint every document and compute pairwise similarities.

    ``docs``: ``{name: text}`` or a list of file paths.
    """
    if isinstance(docs, Mapping):
        names, texts = list(docs), list(docs.values())
    else:
        names = [Path(p).name for p in docs]
        texts = [Path(p).read_text(encoding="utf-8") for p in docs]
    fps = [fingerprint(t, level=level) for t in texts]
    n = len(fps)
    S = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            S[i, j] = S[j, i] = compare(fps[i], fps[j], level=level)
    return DocumentComparison(names, fps, S)
