# Sphota Specification v1, Part 8: Conformance and Certification

```
Status: Frozen procedure (s1). Thresholds may only be raised by RFC, never lowered for s1.
```

## 1. Kinds of conformance

| Subject | Requirement |
|---|---|
| **Code library** (any language) | Round-trips all `tests/vectors/codes.json`; rejects all malformed inputs of `tests/test_codes.py`. |
| **Quantizer implementation** | Given the golden anchor vectors in `tests/vectors/golden.json`, produces exactly the golden codes (default and stable mode). |
| **Encoder variant** (anchor in another runtime/precision) | Certified as *variant* (§3). |
| **Adapter** (another model mapped into s1) | Certified as *certified* or *certified-multilingual* (§3). |

## 2. Official test set and procedure

* **Test set:** FLORES-200 `devtest` (1012 sentences) in 12 languages: en, hi, ta, es, fr, de,
  zh, ar, ru, ja, pt, bn. It is never used to fit the space, the codebooks, the calibration or
  any registry adapter.
* **Reference:** codes from the reference anchor encoder (fp32 ONNX, Part 2 §2).
* **Procedure** (`sphota certify --model <id> [--adapter file.npz]`):
  1. Encode every test sentence with the candidate (model + adapter) and with the reference.
  2. Quantize both with the s1 quantizer (default mode).
  3. Compute, per level `l`:
     * `agreement_same_text_en[l]`: fraction of English sentences whose candidate and
       reference codes share the first `l` levels;
     * `agreement_same_text[l]`: the same over all 12 languages;
     * `agreement_cross_lingual[l]`: fraction of (English, translation) pairs whose
       *candidate* codes share `l` levels (informative);
  4. `fidelity`: mean cosine between candidate canonical vectors and reference canonical
     vectors (English, and all languages).

## 3. Thresholds (s1)

| Grade | Conditions |
|---|---|
| `certified` | `agreement_same_text_en[1] ≥ 0.60`, `agreement_same_text_en[2] ≥ 0.30`, `fidelity_en ≥ 0.75` |
| `certified-multilingual` | `certified` and `agreement_same_text[1] ≥ 0.50`, `agreement_same_text[2] ≥ 0.20`, `fidelity_all ≥ 0.70` |
| `variant` (anchor runtimes) | `agreement_same_text[1] ≥ 0.95` and `agreement_same_text[4] ≥ 0.80` |

*Rationale.* The thresholds are set relative to the anchor's own **cross-lingual** agreement
(L1 ≈ 0.46, L2 ≈ 0.22 on this test set): a certified adapter must agree with the reference on
the *same text* better than the reference agrees with itself across *languages*. A model that
passes therefore adds less disagreement than switching language does.

## 4. Certificate

`sphota certify` prints a certificate and can write it as JSON:

```json
{
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "space": "s1",
  "test_set": "FLORES-200 devtest",
  "agreement_same_text_en": [0.0, 0.0, 0.0, 0.0],
  "agreement_same_text": [0.0, 0.0, 0.0, 0.0],
  "agreement_cross_lingual": [0.0, 0.0, 0.0, 0.0],
  "fidelity_en": 0.0, "fidelity_all": 0.0,
  "checks": {"en_L1": true, "...": true},
  "grade": "certified",
  "badge": "![Sphota s1](https://img.shields.io/badge/...)"
}
```

Measured certificates of the registry adapters are in `docs/benchmark.md` and
`bench/results/adapters-small.json`.

## 5. Re-certification

Certificates name the space version and the test set. A new space version requires new
certificates; a new adapter file requires a new certificate (the registry stores the
adapter's SHA-256 next to its certificate).
