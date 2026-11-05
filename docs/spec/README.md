# Sphota Specification, space version s1

The specification is written in the style of an IETF RFC. Normative key words (MUST,
SHOULD, MAY) follow RFC 2119 / RFC 8174.

| Part | Title | Normative |
|---|---|---|
| [1](01-code-format.md) | Code format: alphabet, levels, widths, permutation, tag, check symbol, grammar, test vectors | yes |
| [2](02-space-and-semantics.md) | Canonical space, anchor, quantization, stable mode, semantics | yes |
| [3](03-confidence-distance-labels.md) | Confidence and precision, alternates, distance and similarity, labels | yes (labels informative) |
| [6](06-versioning-migration.md) | Versioning and migration tables | yes |
| [7](07-adapters-registry.md) | Adapter file format and registry | yes |
| [8](08-conformance.md) | Conformance thresholds and certification procedure | yes |
| [9](09-security-privacy.md) | Security and privacy considerations | informative |
| [10](10-governance.md) | Governance and RFC process | process |

(Parts 4 and 5 were merged into Part 3 during drafting; numbers are kept stable.)

## Summary

```
text ─ anchor (multilingual-e5-small) ─▶ x ∈ R^384
     ─ y = (x − μ)·W  (meaning-LDA, noise-whitened) ─▶ y ∈ R^256
     ─ residual k-means 4096/256/256/256 ─▶ (i1, i2, i3, i4) + calibrated confidence
     ─ affine permutation + Crockford base32 ─▶ "JYZA-7M-X0-A8"
```

Artifacts: `src/sphota/data/v1/` (`manifest.json` lists SHA-256 for every file).
