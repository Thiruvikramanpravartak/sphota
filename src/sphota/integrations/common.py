# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Framework-free helpers shared by the integrations (metadata convention, vector DB payloads).

Metadata / payload convention (use it in any vector database)::

    sphota          full code      'JYZA-7M-X0-A8'
    sphota_l1       level-1 prefix 'JYZA'
    sphota_l2       level-2 prefix 'JYZA-7M'
    sphota_l3       level-3 prefix 'JYZA-7M-X0'
    sphota_label    English label
    sphota_confidence  L1 confidence

Exact-match filters on ``sphota_l1`` / ``sphota_l2`` give prefix filtering in every database
that supports keyword filters (Qdrant, Chroma, Weaviate, Pinecone, Elasticsearch, ...).
"""

from __future__ import annotations

from typing import Any

from sphota.api import Sphota


def tag_metadata(c: Sphota) -> dict[str, Any]:
    d = {
        "sphota": str(c),
        "sphota_label": c.label,
        "sphota_confidence": round(c.confidence[0], 4) if c.confidence else None,
    }
    for lvl in range(1, min(3, c.depth) + 1):
        d[f"sphota_l{lvl}"] = str(c.code.prefix(lvl))
    return d
