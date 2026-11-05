import pytest

TEXTS = [
    "The CEO resigned",
    "El director ejecutivo dimitió",
    "Quantum computers use qubits.",
    "How do I reset my password?",
    "Add two cups of flour and stir until smooth.",
]


def test_langchain(fake_encoder):
    pytest.importorskip("langchain_core")
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever

    from sphota.integrations.langchain import SphotaPrefixRetriever, SphotaTagger

    docs = SphotaTagger().transform_documents(
        [Document(page_content=t, metadata={"i": i}) for i, t in enumerate(TEXTS)]
    )
    assert all("sphota_l1" in d.metadata and d.metadata["i"] == i for i, d in enumerate(docs))

    class Static(BaseRetriever):
        docs: list

        def _get_relevant_documents(self, query, *, run_manager):
            return list(self.docs)

    r = SphotaPrefixRetriever(base=Static(docs=docs), depth=1)
    got = r.invoke("The CEO resigned")
    assert got and all(d.metadata["sphota_l1"] == docs[0].metadata["sphota_l1"] for d in got)
    boost = SphotaPrefixRetriever(
        base=Static(docs=[Document(page_content=t) for t in TEXTS][::-1]), mode="boost", k=2
    )
    assert boost.invoke("The CEO resigned")[0].page_content in TEXTS[:2]


def test_llamaindex(fake_encoder):
    pytest.importorskip("llama_index.core")
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

    from sphota.integrations.llamaindex import (
        SphotaMetadataExtractor,
        SphotaPrefixPostprocessor,
        prefix_filters,
    )

    nodes = SphotaMetadataExtractor()([TextNode(text=t) for t in TEXTS])
    assert all("sphota" in n.metadata for n in nodes)
    pp = SphotaPrefixPostprocessor(depth=1)
    out = pp.postprocess_nodes(
        [NodeWithScore(node=n, score=1.0) for n in nodes], QueryBundle("The CEO resigned")
    )
    assert out and out[0].node.text == TEXTS[0]
    boosted = SphotaPrefixPostprocessor(mode="boost").postprocess_nodes(
        [NodeWithScore(node=TextNode(text=t), score=1.0) for t in TEXTS[::-1]],
        QueryBundle("The CEO resigned"),
    )
    assert boosted[0].node.text in TEXTS[:2]
    f = prefix_filters(nodes[0].metadata["sphota"], depth=2)
    assert f.filters[0].key == "sphota_l2"


def test_qdrant(fake_encoder):
    pytest.importorskip("qdrant_client")
    from qdrant_client import QdrantClient

    from sphota.integrations.qdrant import create_collection, search_in_region, upsert_texts

    client = QdrantClient(":memory:")
    vecs = fake_encoder.encode(TEXTS)
    create_collection(client, "docs", vecs.shape[1])
    codes = upsert_texts(client, "docs", TEXTS, vecs)
    hits = search_in_region(client, "docs", vecs[0], str(codes[0]), depth=1)
    assert hits and hits[0].payload["text"] == TEXTS[0]
    assert all(h.payload["sphota_l1"] == str(codes[0].code.prefix(1)) for h in hits)
