# Sphota Specification v1, Part 10: Governance and the RFC Process

See also [GOVERNANCE.md](https://github.com/thiruvikramanpravartak/sphota/blob/main/GOVERNANCE.md).

## 1. What is normative

Parts 1 (format), 2 (space, quantization), 3 §1-2 and §4 (confidence, distance), 6
(versioning), 7 (adapter format) and 8 (conformance) are normative. Labels, examples and the
neighbour table are informative data shipped with the release.

## 2. Change classes

| Class | Examples | Process |
|---|---|---|
| Editorial | typos, clarifications | PR, one maintainer |
| Additive data | new label languages, new certified adapters | PR, one maintainer + passing certification |
| Library | new APIs, performance | PR, golden tests MUST pass |
| Normative | anything that changes a code for any input; thresholds; formats | RFC → new space version |

## 3. RFC template

```markdown
# RFC NNNN: <title>
- Status: draft | discussion | accepted | rejected | withdrawn
- Authors:
- Space version affected: s1 → s2 | none
## Summary
## Motivation (with measurements)
## Specification (normative text)
## Migration (table, expected agreement s1→s2)
## Alternatives considered
## Security and privacy
## Decision record
```

## 4. Registries

* **Adapter registry**: `adapters/registry.json` (Part 7). Entries need a passing certificate.
* **Space registry**: `docs/spec/spaces.json` lists every published space version with the
  SHA-256 of its manifest. Space versions are never removed.
