# Python API

Everything is one call. Import from `sphota`.

## Encoding

| Call | Returns |
|---|---|
| `encode(text, *, min_confidence=None, max_level=4, stable=False, model=None)` | `Sphota` |
| `encode_batch(texts, ...)` | `list[Sphota]` (batched, cached) |
| `from_vector(vec, *, model=None, ...)` | `Sphota` (1-d) or `list[Sphota]` (2-d) |

`model=None` uses the anchor (ONNX). `model="sentence-transformers/all-MiniLM-L6-v2"` (or a
short name such as `"minilm-l6"`, or a path to an adapter `.npz`) encodes through that
model's certified adapter.

## The `Sphota` object

`Sphota` **is a `str`** (`'JYZA-7M-X0-A8'`): store it, JSON-dump it, use it as a dict key.

| Attribute / method | Meaning |
|---|---|
| `.code` | parsed `Code` (tuple of level indices) |
| `.depth`, `.levels` | number of levels, indices |
| `.label`, `.label_in(lang)` | `'L1 label › L2 label'` |
| `.confidence` | cumulative confidence per level (list of 4) |
| `.level_confidence` | conditional confidence per level |
| `.alternates` | `[(Sphota, p), ...]` other likely codes (same depth) |
| `.candidates(k)` | top-k full codes with probabilities |
| `.is_boundary` | some level has confidence < 0.6 |
| `.prefix(d)`, `.common_prefix(other)` | hierarchy |
| `.similarity(other)`, `.distance(other)`, `.neighbors(k)` | geometry |
| `.vector` | canonical vector (encoded input) or the code's reconstruction |
| `.explain()`, `.to_dict()` | for humans / JSON |

## Codes without a model (numpy only)

`parse(s)`, `is_valid(s)`, `prefix(code, d)`, `similarity(a, b)`, `distance(a, b)`,
`neighbors(code, k)`, `path(a, b)`, `explain(code)`, `label(code, lang)`. When `a`/`b` are
not valid codes they are treated as texts and encoded.

## Documents

| Call | Returns |
|---|---|
| `fingerprint(doc, level=2)` | `Fingerprint` (weighted histogram of prefixes; `.top(n)`) |
| `compare(doc_a, doc_b)` | similarity in [0, 1] |
| `diff(doc_a, doc_b)` | `FingerprintDiff` (`added`, `removed`, `shared`) |

## Use cases (`sphota.usecases`)

`CodeIndex` (search / dedupe / reindex-free migration), `SemanticCache`, `PrefixRouter`,
`analyze_file` / `build_report` / `render_html`, `compare_documents`, `prefix_counts`,
`group_by_prefix`, `cross_lingual_search`, `dedupe`.

## Adapters and conformance (`sphota.adapters`, `sphota.certify`)

`get_adapter(model)`, `load_adapter(path, sha256=None)`, `list_adapters()`,
`train_adapter(model, ...)`, `certify(model, adapter=None)`.

## Errors

`SphotaCodeError` (bad code strings), `EncoderError` (missing extras / offline / download),
`AdapterError` (unknown model, wrong dimension, checksum), `ReleaseError` (corrupted release).
Messages say what to do next.
