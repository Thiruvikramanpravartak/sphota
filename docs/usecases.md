# Use cases

Runnable versions of everything below are in [`examples/`](https://github.com/thiruvikramanpravartak/sphota/tree/main/examples)
(they run in CI).

## 1. Never re-index again (model migration)

```python
from sphota.usecases import CodeIndex

index = CodeIndex()
index.add_texts(docs, model="sentence-transformers/all-MiniLM-L6-v2")  # day 1, model A
index.search("query", model="BAAI/bge-small-en-v1.5")  # day 90, model B
```

Stored items are codes in the frozen s1 space. A new model only needs an adapter (minutes on
a CPU), not a re-embedding of your corpus. Agreement between models is measured per level
(see [Benchmarks](benchmark.md)); use prefix search (L1/L2) plus alternates for recall.

## 2. Topic analytics by prefix

```bash
sphota analyze tickets.csv -o report.html --min-count 10
```

A standalone HTML report: L1 → L2 topic tree, counts, bars, samples and multilingual labels.
In SQL: `SELECT sphota_prefix(code, 1), count(*) FROM tickets GROUP BY 1`.

## 3. Cross-lingual search and dedupe

```python
from sphota.usecases import cross_lingual_search, dedupe

cross_lingual_search("வெள்ளம் நகரத் தெருக்களை மூழ்கடித்தது", docs)  # Tamil query, any-language docs
dedupe(texts, level=3)  # groups sharing 3 levels
```

## 4. Semantic cache across models and languages

```python
from sphota.usecases import SemanticCache

cache = SemanticCache(depth=2, path="cache.db")
answer = cache.get_or_compute(question, call_llm)
```

## 5. Prefix router

```python
from sphota.usecases import PrefixRouter

router = PrefixRouter().fit(example_texts, example_routes)  # or rules={"JYZA": "finance"}
router.route("¿Cómo restablezco mi contraseña?").route
```

## 6. Document meaning fingerprints

```python
from sphota import fingerprint, compare, diff

compare(contract_v1, contract_v2)
diff(contract_v1, contract_v2).added
```

## 7. Share codes instead of text (with honest limits)

Codes let partners compare *topics* without exchanging raw text. They still reveal topics:
share only the depth you need (`max_level=1` or `2`) and aggregate. See [Privacy](privacy.md).
