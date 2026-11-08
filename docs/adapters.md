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

## Nonlinear adapters (set v1.1, the default)

The same five models also have **v1.1** adapters: the linear map plus a small residual network
(one 1024-unit hidden layer), trained on the same pairs (D-028). They roughly double how often the
other model's code lands in the anchor's own cell, keep migration recall at ~100%, and pass the
same certification. Shipped as numpy-only `.npz` (≈ 9.7 MB in total, float16 hidden weights).

```python
encode("The CEO resigned", model="sentence-transformers/all-MiniLM-L6-v2")              # v1.1
encode("The CEO resigned", model="sentence-transformers/all-MiniLM-L6-v2", adapters="v1")  # linear
```

```bash
sphota encode -m minilm-l6 "The CEO resigned"                 # v1.1
SPHOTA_ADAPTERS=v1 python my_app.py                           # the linear set for a process
```

| model | same cell, English: v1 → v1.1 | same cell, 12 languages: v1 → v1.1 | recall@10 (migration) | grade |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | 16.6% → **35.8%** | 5.2% → 8.5% | 99.9% → 99.9% | certified |
| bge-small-en-v1.5 | 13.8% → **36.6%** | 4.9% → 9.8% | 100.0% → 100.0% | certified |
| gte-small | 5.7% → **33.3%** | 3.4% → 9.6% | 99.8% → 100.0% | certified |
| paraphrase-multilingual-MiniLM-L12-v2 | 23.3% → **35.4%** | 24.5% → 32.8% | 100.0% → 100.0% | certified-multilingual |
| LaBSE | 19.2% → **32.8%** | 19.4% → 31.9% | 100.0% → 100.0% | certified-multilingual |

v1.1 has been the default since D-030 (before the first public release, so no stored codes
change). The linear `v1` set stays available, frozen inside the release. Codes from the two sets
can differ for the same model, so record which set produced a stored code. Rebuild:
`python -m sphota.build.adapters_v11 --scale small`.

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
