# Sphota Specification v1, Part 7: Adapter File Format and Registry

```
Status: Frozen format (s1); registry contents grow over time
```

## 1. Adapter

An adapter maps unit-normalized vectors of a source model into the s1 canonical space.

### 1.1 File format

A NumPy `.npz` archive. Readers MUST load it with `allow_pickle=False` and MUST reject any
other container format (pickle, `.pt`, `.bin`). Arrays:

| Key | dtype | shape | Meaning |
|---|---|---|---|
| `A` | float32 | (d_model, D) | linear map |
| `b` | float32 | (D,) | bias |
| `meta` | unicode, 0-d | () | JSON object, see 1.2 |
| `A2`, `b2` | float32 | (D, D), (D,) | optional residual layer (`kind: "mlp"`) |
| `W1`, `c1` | float16 or float32 | (d_model, H), (H,) | optional input network (`kind: "mlp2"`) |
| `W2`, `c2` | float16 or float32 | (H, D), (D,) | optional input network (`kind: "mlp2"`) |

Forward pass: `x̂ = x/‖x‖`, `h = x̂·A + b`; if `A2` present: `y = h + relu(h)·A2 + b2`; if `W1`
present (`kind: "mlp2"`): `y = h + gelu(x̂·W1 + c1)·W2 + c2` with the tanh approximation
`gelu(z) = ½z(1 + tanh(√(2/π)(z + 0.044715 z³)))`, computed in float32; otherwise `y = h`.

### 1.2 `meta` fields

```json
{
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "space": "s1",
  "kind": "linear",
  "prefix": "",
  "pooling": "as model default (sentence-transformers)",
  "train": {"pairs": 98000, "method": "ridge", "alpha": 1.0, "data": ["wikipedia", "flores200-dev", "tatoeba"]},
  "license": "Apache-2.0",
  "model_license": "apache-2.0",
  "created": "2026-11-04"
}
```

## 2. Registry

`adapters/registry.json` inside a release directory:

```json
{
  "space": "s1",
  "adapters": {
    "<hf model id>": {
      "short": "minilm-l6",
      "file": "all-MiniLM-L6-v2.npz",
      "sha256": "<hex>",
      "model_license": "apache-2.0",
      "multilingual": false,
      "certificate": { "...": "see Part 8" }
    }
  }
}
```

* Implementations MUST verify `sha256` before loading an adapter.
* **Adapter sets.** A registry belongs to a set. `v1` is frozen with the release (linear
  adapters). `v1.1` (`sphota/data/adapters/v1.1/`, D-028) holds `mlp2` adapters
  for the same models, certified against the same Part 8 procedure. Sets map into the same
  canonical space; codes from different sets for the same model may differ, so stored codes
  SHOULD record the set (`model@set`). The reference implementation defaults to `v1.1` (D-030); `v1` remains available.
* Only models whose weights are under an OSI-approved (or comparably open) license are
  admitted to the default registry. Adapters for other models (e.g. CC-BY-NC weights or
  proprietary APIs) MAY be trained by users with `sphota train-adapter`; they are not
  distributed.

## 3. Anchor variants

Runtimes or precisions of the anchor itself (int8 ONNX, transformers.js) are registered as
*variants* with an identity-shaped adapter (`A = W`, `b = −μ·W`) and their own certificate.
