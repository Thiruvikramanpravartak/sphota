# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Regions (v1.1): a coarse, universal layer of 32 meaning regions over the s1 canonical space.

Regions are the **cross-language and cross-model** layer: on held-out FLORES data a translation
lands in the same region 70.5% of the time (79.3% when the region confidence is >= 0.7, which holds
for 69% of sentences), and certified models agree with the anchor 64.5% of the time. Finer code
levels (L1-L4) are precision *within* a model and language family. Regions do not change any s1
code; they are an additional, separately versioned artifact (``data/regions``).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

REGIONS_DIR = Path(__file__).parent / "data" / "regions"


@lru_cache(maxsize=1)
def _meta() -> dict:
    return dict(json.loads((REGIONS_DIR / "regions.json").read_text(encoding="utf-8")))


@lru_cache(maxsize=1)
def _load() -> tuple[np.ndarray, float, dict[str, str]]:
    with np.load(REGIONS_DIR / "regions.npz", allow_pickle=False) as z:
        C = z["centroids"].astype(np.float64)
    meta = _meta()
    return C, float(meta["temperature"]), dict(meta["labels"])


def region_of(y: np.ndarray) -> tuple[str, float]:
    """(region id like 'R07', confidence) for one canonical vector."""
    C, T, _ = _load()
    y = np.asarray(y, dtype=np.float64)
    d2 = ((C - y) ** 2).sum(1)
    order = np.argsort(d2)[:16]
    p = np.exp(-(d2[order] - d2[order[0]]) / T)
    p /= p.sum()
    return f"R{int(order[0]):02d}", float(p[0])


def region_label(region: str, lang: str = "en") -> str:
    """Region name in ``lang`` (falls back to English)."""
    key = str(int(region.lstrip("R")))
    if lang != "en":
        tr = _meta().get("labels_i18n", {}).get(lang, {}).get(key)
        if tr:
            return str(tr)
    return _load()[2].get(key, region)


def all_regions() -> dict[str, str]:
    return {f"R{int(k):02d}": v for k, v in _load()[2].items()}
