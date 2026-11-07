# CLAUDE.md: working notes for agents and maintainers

Sphota: readable, multilingual meaning codes for topic analytics, routing, caching and model
migration. A frozen, open, hierarchical code (`7F3A-C2-91-0B`) plus 32 regions for any text in any
language, reachable from many embedding models through adapters. Fine codes are within-model
precision (the GPU gate failed, D-022); do not pitch it as "GPS for meaning" (D-015).

Working memory lives in the repo: **PLAN.md** (milestones), **PROGRESS.md** (status and
measured numbers), **DECISIONS.md** (every non-obvious choice). Re-read them when resuming.

## Architecture

```
text ──encoder──▶ anchor vector (384-d, multilingual-e5-small, "query: " prefix, mean-pool, L2)
                        │  y = (normalize(x) - mu) @ W          (space.npz; PCA + meaning-LDA)
other model ──adapter──▶ canonical vector y (D-d)              (adapters/*.npz, linear)
                        │  residual k-means, 4 levels          (codebooks.npz)
                        ▼
          indices (i1,i2,i3,i4) + per-level confidence          (calibration.json temperatures)
                        │  affine permutation + Crockford base32 (codes.py)
                        ▼
                  "7F3A-C2-91-0B"  + labels (labels.json) + geometry (neighbors.npz)
```

* `src/sphota/codes.py`: code format (pure Python). **Frozen.**
* `src/sphota/release.py`: load + SHA-256-verify release artifacts (numpy).
* `src/sphota/space.py`: projection, quantization, confidence, stable mode (numpy). **Frozen behaviour.**
* `src/sphota/distance.py`: reconstruction, distance, similarity, neighbours, path (numpy).
* `src/sphota/api.py`: public API (`encode`, `Sphota` str subclass, fingerprints, …).
* `src/sphota/encoders.py`: ONNX (default) / torch encoders, HF cache, offline bundle.
* `src/sphota/adapters.py`: adapter format (`.npz` only) + registry.
* `src/sphota/cli.py`: `sphota …` CLI (typer + rich, optional extra).
* `src/sphota/usecases/`: migration, analytics report, dedupe/search, cache, router.
* `src/sphota/integrations/`: LangChain, LlamaIndex, Postgres SQL, Qdrant/Chroma.
* `src/sphota/build/`: offline pipeline (data, embed, fit, bake-off, release, adapters, bench).
* `src/sphota/data/v1/`: **the frozen release** (do not edit by hand; rebuilt only via RFC).
* `atlas/`: Meaning Atlas (Vite + TS, transformers.js). `docs/`: spec, mkdocs site, paper.

## Commands

```bash
source .venv/bin/activate                     # uv venv with [dev] extras
pytest -m "not slow"                          # fast tests (no network)
pytest -m slow                                # model-download tests
ruff check . && ruff format --check . && mypy
python -m sphota.build.data --scale small     # fetch data (cached in data/)
python -m sphota.build.release --scale small  # rebuild release (changes the standard! RFC only)
python -m sphota.build.adapters --scale small # train + certify registry adapters
python -m sphota.build.bench --scale small    # benchmark → bench/results/*.json
cd atlas && npm ci && npm run build           # Meaning Atlas
mkdocs build                                  # docs site
```

## Conventions

* Every script takes `--scale {tiny,small,full}`. `small` is what this repo ships; `full` is
  for a GPU.
* Light core = numpy only; anything needing torch/onnxruntime imports lazily.
* Error messages say what to do next.
* No pickles anywhere. Artifacts are `.npz` (allow_pickle=False) or `.json`, checksummed.
* No files > 50 MB in git; models download from the Hub.
* Branch: `claude/eloquent-johnson-kla3lt` (see DECISIONS D-001).
