# Integrations

All integrations use one metadata/payload convention, so codes written by one tool can be
queried by any other:

| key | example | use |
|---|---|---|
| `sphota` | `JYZA-7M-X0-A8` | full code |
| `sphota_l1` | `JYZA` | broad topic: filter / group by |
| `sphota_l2` | `JYZA-7M` | sub-topic: filter / group by |
| `sphota_l3` | `JYZA-7M-X0` | narrow theme |
| `sphota_label` | `Corporate Leadership Changes` | display |
| `sphota_confidence` | `0.80` | L1 confidence |

## PostgreSQL (pure SQL)

```bash
docker compose -f integrations/postgres/docker-compose.yml up -d
psql postgresql://postgres:sphota@localhost/sphota
```

```sql
SELECT body, code FROM docs WHERE code LIKE sphota_like('JYZA-7M');      -- index scan
SELECT sphota_prefix(code, 1) AS topic, count(*) FROM docs GROUP BY 1;    -- analytics
SELECT body, sphota_similarity(code, 'JYZA-7M-X0-A8') AS s FROM docs ORDER BY s DESC LIMIT 5;
```

Functions: `sphota_is_code`, `sphota_depth`, `sphota_prefix`, `sphota_level`, `sphota_like`,
`sphota_is_prefix`, `sphota_common_prefix`, `sphota_similarity` (uses the shipped L1
neighbour table), `sphota_nearby_l1`; domain `sphota_code`. From Python:
`sphota.integrations.postgres.install(conn)`. Tested against PostgreSQL 16
(`tests/test_postgres.py`).

## LangChain

```python
from sphota.integrations.langchain import SphotaTagger, SphotaPrefixRetriever

docs = SphotaTagger().transform_documents(docs)  # adds metadata
retriever = SphotaPrefixRetriever(base=store.as_retriever(), depth=1)  # same-topic answers only
```

## LlamaIndex

```python
from sphota.integrations.llamaindex import (
    SphotaMetadataExtractor,
    SphotaPrefixPostprocessor,
    prefix_filters,
)

pipeline = IngestionPipeline(transformations=[splitter, SphotaMetadataExtractor()])
engine = index.as_query_engine(node_postprocessors=[SphotaPrefixPostprocessor(depth=1)])
retriever = index.as_retriever(filters=prefix_filters("JYZA-7M", depth=2))
```

## Qdrant (and any vector DB with keyword filters)

```python
from qdrant_client import QdrantClient
from sphota.integrations.qdrant import create_collection, upsert_texts, search_in_region

client = QdrantClient(":memory:")
create_collection(client, "docs", dim=384)
codes = upsert_texts(client, "docs", texts, vectors)
hits = search_in_region(client, "docs", query_vector, code="JYZA", depth=1)
```

Chroma / Weaviate / Pinecone / Elasticsearch: store the same payload keys and use an
exact-match filter on `sphota_l1` / `sphota_l2`.
