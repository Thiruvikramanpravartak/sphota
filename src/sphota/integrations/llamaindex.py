# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""LlamaIndex integration (needs ``llama-index-core``).

* :class:`SphotaMetadataExtractor`: a ``TransformComponent`` that adds Sphota metadata to
  nodes during ingestion.
* :class:`SphotaPrefixPostprocessor`: node postprocessor that keeps (or boosts) retrieved
  nodes sharing a code prefix with the query.
* :func:`prefix_filters`: ``MetadataFilters`` for vector stores that filter on metadata
  (``sphota_l1 == 'JYZA'``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

try:
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle, TransformComponent
    from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
except ImportError as e:  # pragma: no cover
    raise ImportError("the LlamaIndex integration needs: pip install llama-index-core") from e

from sphota.api import Sphota, encode, encode_batch
from sphota.integrations.common import tag_metadata


class SphotaMetadataExtractor(TransformComponent):
    model: str | None = None
    max_level: int = 4

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        codes = encode_batch(
            [n.get_content() for n in nodes], model=self.model, max_level=self.max_level
        )
        for n, c in zip(nodes, codes):
            n.metadata.update(tag_metadata(c))
        return nodes


class SphotaPrefixPostprocessor(BaseNodePostprocessor):
    depth: int = 1
    mode: str = "filter"
    model: str | None = None

    def _postprocess_nodes(
        self, nodes: list[NodeWithScore], query_bundle: QueryBundle | None = None
    ) -> list[NodeWithScore]:
        if query_bundle is None:
            return nodes
        q = encode(query_bundle.query_str, model=self.model)
        scored = []
        for i, n in enumerate(nodes):
            code = n.node.metadata.get("sphota")
            if code is None:
                code = str(encode(n.node.get_content(), model=self.model))
                n.node.metadata.update({"sphota": code})
            scored.append((q.code.common_prefix(Sphota(code).code), i, n))
        if self.mode == "filter":
            return [n for s, _, n in scored if s >= self.depth]
        return [n for _, _, n in sorted(scored, key=lambda t: (-t[0], t[1]))]


def prefix_filters(code: str, depth: int = 1) -> MetadataFilters:
    c = Sphota(code)
    key = {1: "sphota_l1", 2: "sphota_l2"}.get(depth, "sphota")
    return MetadataFilters(
        filters=[ExactMatchFilter(key=key, value=str(c.code.prefix(min(depth, c.depth))))]
    )
