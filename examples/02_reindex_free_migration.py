"""Never re-index again: index with model A, query with model B, through the shared space.

We index documents with all-MiniLM-L6-v2 (via its certified adapter) and later query with the
Sphota anchor model (and with bge-small-en-v1.5): no re-embedding of the stored documents.
"""

import csv
from pathlib import Path

from sphota.usecases import CodeIndex

rows = list(
    csv.DictReader(open(Path(__file__).parent / "data" / "headlines.csv", encoding="utf-8"))
)
english = [r for r in rows if r["lang"] == "en"]

# Day 1: index with model A (codes only are stored; vectors can be thrown away)
index = CodeIndex()
index.add_texts(
    [r["text"] for r in english],
    ids=[r["id"] for r in english],
    model="sentence-transformers/all-MiniLM-L6-v2",
)
print(f"indexed {len(index)} docs with all-MiniLM-L6-v2 → codes")

# Day 90: we switched models. Query with the anchor and with bge-small: no re-indexing.
queries = {
    "business": "Stock prices dropped after the earnings announcement",
    "weather": "Streets were flooded by torrential rain",
    "science": "Scientists found a planet with water",
}
for model in [None, "BAAI/bge-small-en-v1.5"]:
    hits_ok = 0
    for topic, q in queries.items():
        hits = index.search(q, k=3, model=model)
        top = next(r for r in english if r["id"] == hits[0].id)
        hits_ok += top["topic"] == topic
        print(
            f"[{model or 'anchor'}] {q!r} → {top['text'][:60]!r} ({hits[0].code}, {hits[0].score:.2f})"
        )
    print(f"  top-1 topic accuracy with {model or 'anchor'}: {hits_ok}/{len(queries)}")
