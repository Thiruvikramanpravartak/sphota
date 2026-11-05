# Progress

_Last updated: during M4 (adapter quality investigation)._

## Done
- **R**: licenses verified (docs/LICENSES-THIRD-PARTY.md), tiny bake-off (4 anchors × 3 quantizers),
  gap analysis, design sweep at small scale. Decisions D-001…D-011.
- **M0**: layout, pyproject (hatch), ruff, mypy (core), pytest, pre-commit, CI workflow, governance files.
- **M1**: spec parts 1, 2, 3, 6, 7, 8, 9, 10 (docs/spec/).
- **M2 (first build)**: s1 release built at small scale (src/sphota/data/v1), golden vectors.
- **M3**: SDK (encode/batch/from_vector/explain/similarity/neighbors/path/alternates/candidates/
  fingerprint/compare/diff), ONNX encoder (bit-exact with torch), CLI, tests (fast suite green).
- **M6**: use cases (CodeIndex, SemanticCache, PrefixRouter, analytics HTML report, documents).
- **M7**: Postgres SQL helpers (tested on a real PG16 cluster), LangChain, LlamaIndex, Qdrant.
- **M8 (partial)**: Atlas app code (Vite + TS + transformers.js); data export script.
- Docs drafts, examples 01–08, Colab notebook, HF Space app, launch kit drafts.

## Doing
- **M4 adapter quality.** First linear adapter (all-MiniLM-L6-v2) agrees with the anchor at L1
  only 16.6% (fidelity cos 0.55). Diagnosis: the canonical space is whitened against *e5's own*
  translation noise, amplifying e5-specific directions. Experiment (`xmodel.py` in scratch,
  results in data/xmodel*.log): add cross-model residual scatter to S_within (λ), hold out
  bge-small to test generalization. If it helps, **rebuild s1** (allowed: nothing published yet),
  regenerate golden vectors, labels (LLM naming cache is per-cell, will be invalidated),
  adapters, benchmark, atlas data.

## Next
- M4 finish (adapters for 6 models, vec2vec run), M5 benchmark + docs/benchmark.md + leaderboard,
  labels (LLM + MT), atlas data + build + screenshot, M9 open data, M10 README/paper/launch,
  M11 hardening, M12 audit.

## Blockers / incidents
- Container restart once (background jobs killed; all caches resumable).
- CPU contention: never run two torch jobs with 4 threads each (use SPHOTA_THREADS=2).

## Measured numbers (small scale, FLORES devtest, 12 languages)
- Cross-lingual L1/L2/L3/L4 agreement (en↔x), first s1 build: 0.46 / 0.22 / 0.10 / 0.05;
  PCA-only baseline 0.34 / 0.08.
- Utilization: 92% of 4096 L1 codes used on held-out Wikipedia.
- ONNX fp32 vs torch: max abs diff 7e-8; int8 cosine 0.997.
