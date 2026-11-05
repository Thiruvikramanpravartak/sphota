"""Cross-lingual search and near-duplicate detection with codes only."""

import csv
from collections import defaultdict
from pathlib import Path

from sphota.usecases import CodeIndex

rows = list(
    csv.DictReader(open(Path(__file__).parent / "data" / "headlines.csv", encoding="utf-8"))
)
index = CodeIndex()
index.add_texts([r["text"] for r in rows], ids=[r["id"] for r in rows])

# A Tamil query finds Hindi, Spanish, English... documents
for hit in index.search("வெள்ளம் நகரத் தெருக்களை மூழ்கடித்தது", k=4):  # "flood swamped city streets"
    r = next(x for x in rows if x["id"] == hit.id)
    print(f"{hit.score:.2f} [{r['lang']}] {r['text'][:70]}")

# Translations of the same sentence often share the full code or a long prefix
groups = defaultdict(list)
for lvl in (3, 2):
    for g in index.duplicates(level=lvl):
        langs = sorted({next(x for x in rows if x["id"] == i)["lang"] for i in g})
        if len(langs) > 1:
            groups[lvl].append(g)
    print(f"cross-lingual groups sharing {lvl} levels: {len(groups[lvl])}")
