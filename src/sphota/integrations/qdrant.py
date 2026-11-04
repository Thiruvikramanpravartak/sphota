# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""Qdrant example helpers (needs ``qdrant-client``): payloads with Sphota prefixes and
prefix-filtered search. Works with ``QdrantClient(":memory:")`` for local experiments.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

try:
    from qdrant_client import QdrantClient, models
except ImportError as e:  # pragma: no cover
    raise ImportError("the Qdrant helpers need: pip install qdrant-client") from e

from sphota.api import Sphota, encode_batch
from sphota.integrations.common import tag_metadata


def create_collection(client: QdrantClient, name: str, dim: int) -> None:
    client.create_collection(
        name, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
    )
    for field_name in ["sphota_l1", "sphota_l2"]:
        client.create_payload_index(
            name, field_name=field_name, field_schema=models.PayloadSchemaType.KEYWORD
        )


def upsert_texts(
    client: QdrantClient,
    name: str,
    texts: Sequence[str],
    vectors: np.ndarray,
    payloads: Sequence[dict[str, Any]] | None = None,
    codes: Sequence[Sphota] | None = None,
) -> list[Sphota]:
    codes = list(codes) if codes is not None else encode_batch(list(texts))
    points = []
    for i, (t, v, c) in enumerate(zip(texts, vectors, codes)):
        payload = {"text": t, **tag_metadata(c), **(payloads[i] if payloads else {})}
        points.append(models.PointStruct(id=i, vector=[float(x) for x in v], payload=payload))
    client.upsert(name, points=points)
    return codes


def prefix_filter(code: str, depth: int = 1) -> models.Filter:
    c = Sphota(code)
    key = {1: "sphota_l1", 2: "sphota_l2"}[min(depth, 2)]
    return models.Filter(
        must=[
            models.FieldCondition(
                key=key, match=models.MatchValue(value=str(c.code.prefix(min(depth, c.depth))))
            )
        ]
    )


def search_in_region(
    client: QdrantClient,
    name: str,
    query_vector: np.ndarray,
    code: str,
    depth: int = 1,
    limit: int = 5,
) -> list[Any]:
    res = client.query_points(
        name,
        query=[float(x) for x in query_vector],
        query_filter=prefix_filter(code, depth),
        limit=limit,
    )
    return list(res.points)
