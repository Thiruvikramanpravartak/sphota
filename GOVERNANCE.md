# Governance

Sphota is a *standard* as much as a library. Governance protects the one promise that makes
it useful: **a v1 code means the same thing forever.**

## Roles

* **Maintainers** review and merge PRs, cut releases, and run the RFC process.
* **Spec editors** (a subset of maintainers) own `docs/spec/` and the frozen release files.
* **Contributors**: everyone else. Anyone may write an RFC.

## What can change without an RFC

* Library code, docs, adapters, integrations, the Atlas, as long as golden tests pass.
* New labels in additional languages (labels are *additive*; an existing label string may
  be corrected for typos but not re-pointed to another meaning).
* New certified adapters in the registry.

## What needs an RFC

* Anything that changes the code assigned to a vector: projection, codebooks, quantization
  rule, tie-breaking. These ship only as a **new space version** (`s2`), never as a v1 patch.
* Changes to the code format, conformance thresholds or adapter/registry file formats.

## RFC process

1. Open a PR adding `docs/rfcs/NNNN-title.md` (template in `docs/spec/10-governance.md`).
2. Discussion for at least 14 days; maintainers summarize objections.
3. Decision by maintainer consensus (lazy consensus; any spec editor may block with a
   written rationale). Decisions are recorded in the RFC file.
4. Accepted space changes are released as a new version with a `vN → vN+1` migration table
   produced by `sphota migrate --build`.

## Decision log

Engineering decisions are logged in [DECISIONS.md](DECISIONS.md).
