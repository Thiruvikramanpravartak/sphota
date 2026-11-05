# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Semantic cache keyed by Sphota code prefixes.

Works across languages (a Hindi question can hit an English cache entry) and across
embedding models (queries encoded by any certified adapter share keys). The key depth sets
the precision/recall trade-off: depth 2 = "same sub-topic", depth 4 = "near-paraphrase".
``min_confidence`` avoids caching answers under codes the encoder is unsure about.

Backends: in-memory dict (default) or SQLite (persistent, ``path=...``).
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sphota.api import Sphota, encode


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    skipped_low_confidence: int = 0

    @property
    def hit_rate(self) -> float:
        n = self.hits + self.misses
        return self.hits / n if n else 0.0


@dataclass
class SemanticCache:
    depth: int = 3
    min_confidence: float = 0.0
    model: str | None = None
    path: str | Path | None = None
    ttl_seconds: float | None = None
    probe_alternates: bool = False
    stats: CacheStats = field(default_factory=CacheStats)

    def __post_init__(self) -> None:
        self._mem: dict[str, tuple[float, Any]] = {}
        self._db: sqlite3.Connection | None = None
        if self.path is not None:
            self._db = sqlite3.connect(str(self.path))
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS sphota_cache (key TEXT PRIMARY KEY, t REAL, value TEXT)"
            )

    def key(self, text: str) -> tuple[str, Sphota]:
        c = encode(text, model=self.model)
        return str(c.code.prefix(self.depth)), c

    def _get(self, k: str) -> Any | None:
        if self._db is not None:
            row = self._db.execute(
                "SELECT t, value FROM sphota_cache WHERE key = ?", (k,)
            ).fetchone()
            hit = (row[0], json.loads(row[1])) if row else None
        else:
            hit = self._mem.get(k)
        if hit is None:
            return None
        t, v = hit
        if self.ttl_seconds is not None and time.time() - t > self.ttl_seconds:
            return None
        return v

    def _put(self, k: str, v: Any) -> None:
        if self._db is not None:
            self._db.execute(
                "INSERT OR REPLACE INTO sphota_cache VALUES (?, ?, ?)",
                (k, time.time(), json.dumps(v)),
            )
            self._db.commit()
        else:
            self._mem[k] = (time.time(), v)

    def get(self, text: str) -> Any | None:
        k, c = self.key(text)
        v = self._get(k)
        if v is None and self.probe_alternates:
            for alt, _ in c.alternates[:2]:
                v = self._get(str(alt.code.prefix(min(self.depth, alt.depth))))
                if v is not None:
                    break
        if v is None:
            self.stats.misses += 1
        else:
            self.stats.hits += 1
        return v

    def set(self, text: str, value: Any) -> bool:
        k, c = self.key(text)
        if c.confidence[self.depth - 1] < self.min_confidence:
            self.stats.skipped_low_confidence += 1
            return False
        self._put(k, value)
        return True

    def get_or_compute(self, text: str, fn: Callable[[str], Any]) -> Any:
        v = self.get(text)
        if v is None:
            v = fn(text)
            self.set(text, v)
        return v

    def __len__(self) -> int:
        if self._db is not None:
            return int(self._db.execute("SELECT COUNT(*) FROM sphota_cache").fetchone()[0])
        return len(self._mem)
