# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""A tiny code index: store items once as Sphota codes, query from any model or language.

This is the "never re-index again" use case. Items indexed with model A (through its
adapter) can be queried with model B (through its adapter), or with the anchor, because
both land in the same frozen space.

Two storage modes:

* ``store_vectors=True`` (default): keep each item's 256-d **canonical vector** (the
  model-independent point in the s1 space) next to its code. Queries from any certified
  model are ranked by cosine in the canonical space; this is what makes cross-model search
  precise (benchmark: 97-99% recall@10 for paraphrase retrieval across models). Codes are
  used for region filters (``within=``) and are what you show, group and share.
* ``store_vectors=False``: codes only (13 characters per item). Candidates come from the
  query's L1 cell, its top alternates and neighbouring cells; ranking uses hierarchy-aware
  code similarity. Good for topic-level grouping, routing and dedupe; coarse for ranking.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sphota.api import Sphota, encode_batch, from_vector
from sphota.codes import Code
from sphota.distance import similarity


@dataclass
class Hit:
    id: Any
    code: str
    score: float
    item: Any = None

    def __repr__(self) -> str:
        return f"Hit(id={self.id!r}, code={self.code}, score={self.score:.3f})"


@dataclass
class CodeIndex:
    """In-memory prefix index. Keys are codes (strings); values are your item ids/payloads."""

    probes: int = 3
    store_vectors: bool = True
    _codes: dict[Any, Code] = field(default_factory=dict)
    _vecs: dict[Any, np.ndarray] = field(default_factory=dict)
    _items: dict[Any, Any] = field(default_factory=dict)
    _by_l1: dict[int, list[Any]] = field(default_factory=lambda: defaultdict(list))

    def __len__(self) -> int:
        return len(self._codes)

    # ---- adding
    def add_codes(
        self, ids: Sequence[Any], codes: Sequence[str | Sphota], items: Sequence[Any] | None = None
    ) -> None:
        from sphota.codes import parse

        for j, (i, c) in enumerate(zip(ids, codes)):
            code = c.code if isinstance(c, Sphota) else parse(c)
            self._codes[i] = code
            self._by_l1[code.levels[0]].append(i)
            if self.store_vectors and isinstance(c, Sphota) and c._vector is not None:
                self._vecs[i] = np.asarray(c._vector, dtype=np.float32)
            if items is not None:
                self._items[i] = items[j]

    def add_texts(
        self,
        texts: Sequence[str],
        ids: Sequence[Any] | None = None,
        model: str | None = None,
        store_text: bool = True,
    ) -> list[Sphota]:
        ids = list(ids) if ids is not None else list(range(len(self), len(self) + len(texts)))
        codes = encode_batch(list(texts), model=model)
        self.add_codes(ids, codes, list(texts) if store_text else None)
        return codes

    def add_vectors(
        self,
        vectors: np.ndarray,
        ids: Sequence[Any],
        model: str | None = None,
        items: Sequence[Any] | None = None,
    ) -> list[Sphota]:
        codes = from_vector(np.atleast_2d(vectors), model=model)
        assert isinstance(codes, list)
        self.add_codes(ids, codes, items)
        return codes

    def remove(self, id_: Any) -> None:
        code = self._codes.pop(id_)
        self._by_l1[code.levels[0]].remove(id_)
        self._items.pop(id_, None)
        self._vecs.pop(id_, None)

    # ---- querying
    def _candidates(self, q: Sphota) -> list[tuple[Any, float]]:
        l1s: list[tuple[int, float]] = [(q.code.levels[0], 1.0)]
        for alt, _p in q.alternates[: self.probes - 1]:
            if alt.code.levels[0] not in [x for x, _ in l1s]:
                l1s.append((alt.code.levels[0], 0.95))
        out = []
        for l1, w in l1s:
            for i in self._by_l1.get(l1, []):
                out.append((i, w))
        return out

    def search_code(
        self, q: Sphota, k: int = 10, expand: bool = True, within: str | None = None
    ) -> list[Hit]:
        """Rank items for query ``q``. ``within``: only items under this code prefix."""
        if within is not None:
            from sphota.codes import parse

            region = parse(within)
            allowed = [i for i, c in self._codes.items() if region.is_prefix_of(c)]
        else:
            allowed = None
        if self._vecs and q._vector is not None and len(self._vecs) == len(self._codes):
            ids = list(self._vecs) if allowed is None else allowed
            if not ids:
                return []
            M = np.stack([self._vecs[i] for i in ids])
            qv = np.asarray(q._vector, dtype=np.float32)
            sims = (M @ qv) / (np.linalg.norm(M, axis=1) * np.linalg.norm(qv) + 1e-12)
            order = np.argsort(-sims)[:k]
            return [
                Hit(ids[j], str(self._codes[ids[j]]), float(sims[j]), self._items.get(ids[j]))
                for j in order
            ]
        cands = self._candidates(q)
        if allowed is not None:
            keep = set(allowed)
            cands = [(i, w) for i, w in cands if i in keep]
            expand = False
        seen: dict[Any, float] = {}
        for i, w in cands:
            s = w * similarity(q.code, self._codes[i])
            seen[i] = max(seen.get(i, 0.0), s)
        if expand and len(seen) < k:  # widen to neighbouring L1 regions
            from sphota.distance import neighbors

            for nb, _ in neighbors(q.code.prefix(1), 8):
                for i in self._by_l1.get(nb.levels[0], []):
                    if i not in seen:
                        seen[i] = 0.9 * similarity(q.code, self._codes[i])
        ranked = sorted(seen.items(), key=lambda kv: -kv[1])[:k]
        return [Hit(i, str(self._codes[i]), s, self._items.get(i)) for i, s in ranked]

    def search(
        self, text: str, k: int = 10, model: str | None = None, within: str | None = None
    ) -> list[Hit]:
        """Search with a text (any language), optionally encoded by another model."""
        q = encode_batch([text], model=model)[0]
        return self.search_code(q, k, within=within)

    def search_vector(self, vec: np.ndarray, k: int = 10, model: str | None = None) -> list[Hit]:
        q = from_vector(np.asarray(vec), model=model)
        assert isinstance(q, Sphota)
        return self.search_code(q, k)

    # ---- dedupe
    def duplicates(self, level: int = 4) -> list[list[Any]]:
        """Groups of ids sharing the same code prefix at ``level`` (size >= 2)."""
        groups: dict[Code, list[Any]] = defaultdict(list)
        for i, c in self._codes.items():
            if c.depth >= level:
                groups[c.prefix(level)].append(i)
        return [g for g in groups.values() if len(g) > 1]

    def codes(self) -> dict[Any, str]:
        return {i: str(c) for i, c in self._codes.items()}


def dedupe(texts: Sequence[str], level: int = 4, model: str | None = None) -> list[list[int]]:
    """Cross-lingual near-duplicate groups (indices into ``texts``) by shared code prefix."""
    idx = CodeIndex()
    idx.add_texts(texts, model=model, store_text=False)
    return idx.duplicates(level)


def cross_lingual_search(query: str, docs: Iterable[str], k: int = 5) -> list[Hit]:
    """One-liner: search ``docs`` (any languages) with ``query`` (any language)."""
    idx = CodeIndex()
    idx.add_texts(list(docs))
    return idx.search(query, k)
