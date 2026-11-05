# Adapters

An adapter maps another embedding model into the frozen s1 space. It is a small linear map
(`.npz`, no code, SHA-256 checksummed) fitted on 40k texts embedded by both the model and the
anchor. Registry adapters run through ONNX Runtime (no torch) when an ONNX export exists.

```python
from sphota import encode, from_vector

encode("The CEO resigned", model="sentence-transformers/all-MiniLM-L6-v2")
from_vector(my_bge_vectors, model="BAAI/bge-small-en-v1.5")  # vectors you already have
```

```bash
sphota adapters                                   # list certified adapters
sphota train-adapter --model intfloat/e5-small-v2 --prefix "query: "
sphota certify --model intfloat/e5-small-v2 --adapter e5-small-v2.sphota-adapter.npz --out cert.json
```

## Registry (space s1)

| model | license | certificate |
|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | Apache-2.0 | certified |
| `BAAI/bge-small-en-v1.5` | MIT | certified |
| `thenlper/gte-small` | MIT | certified |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Apache-2.0 | certified-multilingual |
| `sentence-transformers/LaBSE` | Apache-2.0 | certified-multilingual (torch; no small ONNX export) |

`BAAI/bge-m3` was trained and evaluated at a CPU-limited budget (6k pairs) and missed the
neighbourhood threshold (0.53 < 0.55); it is not in the registry. Its codes were the most
cross-lingually consistent of all models (65% L1), so it is the first candidate for a
full-scale adapter.

Full certificates (exact and neighbourhood agreement, fidelity) are in the
[benchmark](benchmark.md) and the [leaderboard](https://github.com/thiruvikramanpravartak/sphota/blob/main/bench/leaderboard/index.html).

## What certification means

Certification in s1 guarantees **neighbourhood-level interoperability**: the model places most
texts in the anchor's L1 region or one of its 8 nearest regions. It does not guarantee
identical codes (exact L1 agreement is 6–23% for small open models). For precise cross-model
retrieval, store the canonical vector (`c.vector`, 256 floats) next to the code:
`CodeIndex(store_vectors=True)` does this and reaches 96–99% recall@10 across models.

## Per-adapter confidence

Each adapter carries a `temperature_scale` fitted on held-out pairs, so confidence for a noisier
model is wider and `min_confidence` truncates more, instead of being over-confident.

## Proprietary APIs and non-commercial models

Recipes for OpenAI and Cohere embeddings live in `adapters/recipes/` (your key, your adapter; we
do not distribute adapters trained on API outputs). Models with non-commercial weights (e.g.
jina-embeddings-v3) can be adapted locally with `sphota train-adapter` but are not accepted into
the default registry.

## Unsupervised adapters (experiment)

A vec2vec-style adapter trained without paired data is reported in the benchmark; at small scale
it failed completely (top-1 retrieval 0% vs 96.5% supervised). Paired data from the anchor is cheap to produce, so
supervised adapters remain the default.
