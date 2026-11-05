# Copyright 2026 The Sphota Authors
# SPDX-License-Identifier: Apache-2.0
"""LangChain integration (needs ``langchain-core``).

* :class:`SphotaTagger`: document transformer that adds ``sphota`` metadata
  (``sphota``, ``sphota_l1``, ``sphota_l2``, ``sphota_label``, ``sphota_confidence``).
* :class:`SphotaPrefixRetriever`: wraps any retriever; keeps (or boosts) documents whose code
  shares a prefix with the query's code, e.g. "only answer from the same topic region".

>>> from sphota.integrations.langchain import SphotaTagger, SphotaPrefixRetriever  # doctest: +SKIP
>>> docs = SphotaTagger().transform_documents(docs)                                # doctest: +SKIP
>>> retriever = SphotaPrefixRetriever(base=vectorstore.as_retriever(), depth=1)     # doctest: +SKIP
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

try:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import BaseDocumentTransformer, Document
    from langchain_core.retrievers import BaseRetriever
except ImportError as e:  # pragma: no cover
    raise ImportError("the LangChain integration needs: pip install langchain-core") from e

from sphota.api import Sphota, encode, encode_batch
from sphota.integrations.common import tag_metadata


class SphotaTagger(BaseDocumentTransformer):
    """Add Sphota codes to ``Document.metadata`` (batched)."""

    def __init__(
        self, model: str | None = None, max_level: int = 4, min_confidence: float | None = None
    ):
        self.model = model
        self.max_level = max_level
        self.min_confidence = min_confidence

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        codes = encode_batch(
            [d.page_content for d in documents],
            model=self.model,
            max_level=self.max_level,
            min_confidence=self.min_confidence,
        )
        out = []
        for d, c in zip(documents, codes):
            out.append(
                Document(page_content=d.page_content, metadata={**d.metadata, **tag_metadata(c)})
            )
        return out


class SphotaPrefixRetriever(BaseRetriever):
    """Filter (``mode='filter'``) or re-rank (``mode='boost'``) a base retriever's results by
    the shared code prefix with the query. Documents must carry ``metadata['sphota']``
    (see :class:`SphotaTagger`); untagged documents are tagged on the fly."""

    base: BaseRetriever
    depth: int = 1
    mode: str = "filter"
    model: str | None = None
    k: int | None = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        docs = self.base.invoke(query)
        q = encode(query, model=self.model)
        untagged = [d for d in docs if "sphota" not in d.metadata]
        if untagged:
            for d, c in zip(
                untagged, encode_batch([d.page_content for d in untagged], model=self.model)
            ):
                d.metadata.update(tag_metadata(c))
        scored = [
            (q.code.common_prefix(Sphota(d.metadata["sphota"]).code), i, d)
            for i, d in enumerate(docs)
        ]
        if self.mode == "filter":
            kept = [d for s, _, d in scored if s >= self.depth]
        else:
            kept = [d for _, _, d in sorted(scored, key=lambda t: (-t[0], t[1]))]
        return kept[: self.k] if self.k else kept
