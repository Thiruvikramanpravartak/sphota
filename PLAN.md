# Sphota v1 build plan

Status legend: `[x]` done, `[~]` partial (see PROGRESS.md), `[ ]` todo.

## R: Research and validation sprint
- [x] Verify model and dataset licenses → `docs/LICENSES-THIRD-PARTY.md`
- [x] Bake-off at `tiny` scale: 4 anchors × 3 quantizers × {PCA, aligned} → `bench/results/bakeoff-tiny.json`
- [x] Gap analysis → `docs/research/gap-analysis.md`
- [x] Choose anchor, quantizer, dims; log in DECISIONS.md

### Refined design (after R)
- Anchor: `intfloat/multilingual-e5-small` (MIT, 384-d, ONNX available, runs in the browser).
- Canonical space: L2-normalize → center → PCA → *meaning-LDA* (noise-whitened w.r.t.
  translation scatter, FLORES dev + Tatoeba) → D = 256.
- Quantizer: residual k-means, sizes 4096/256/256/256 (k-means++ init; 92% of L1 cells used).
- Cross-model noise term in the space: tried, no gain at 4096 cells (D-012); certification is
  neighbourhood-based (D-013); never-re-index uses stored canonical vectors + codes.
- Confidence: softmax over the top-16 centroid distances with a per-level temperature fitted by
  maximum likelihood on held-out translation pairs (proper scoring rule ⇒ calibrated).
- Stable mode: majority vote of nearest centroid under fixed Gaussian perturbations.
- Code: Crockford base32 `XXXX-XX-XX-XX`, per-level affine symbol permutation (sparse code
  space, typo detection), optional `s1:` tag and `.X` mod-37 check symbol.

## M0: Foundation
- [x] Layout, pyproject (hatch), ruff, mypy (core), pytest, pre-commit, CI
- [x] CLAUDE.md, PLAN.md, PROGRESS.md, DECISIONS.md
- [x] CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, GOVERNANCE, CHANGELOG, NOTICE

## M1: Specification v1 (docs/spec/)
- [x] Code format, alphabet, levels, semantics
- [x] Confidence / precision semantics; distance; labels
- [x] Versioning & migration; conformance thresholds & procedure
- [x] Adapter file format; registry format; governance/RFC; security & privacy

## M2: Core coordinate space v1
- [x] Small-scale data + anchor embeddings
- [x] Canonical space + alignment (measure before/after)
- [x] Quantizer, calibration, stable mode
- [x] Labels (multilingual keyphrases), distance/neighbour table
- [x] Freeze: manifest + checksums in `src/sphota/data/v1`

## M3: SDK + CLI
- [x] encode / encode_batch / explain / similarity / prefix / neighbors / path / alternates
- [x] fingerprint / compare / diff / from_vector
- [x] ONNX encoder (default), torch optional, cache, batch
- [x] CLI mirrors API; coverage ≥ 85% core; fresh-venv install test

## M4: Adapters + registry
- [x] ≥4 adapters (all-MiniLM-L6-v2, bge-small-en-v1.5, gte-small, multilingual MiniLM, LaBSE, bge-m3)
- [x] vec2vec-style unsupervised experiment (honest report)
- [x] `sphota train-adapter`, `sphota certify`; OpenAI/Cohere recipes (not run)

## M5: Benchmark + leaderboard
- [x] Official test set; agreement per level (cross-lingual, cross-model); paraphrase stability
- [x] Utilization; retrieval with codes vs raw; truncation + stable mode effects
- [x] `bench/results/*.json`, `docs/benchmark.md`, static leaderboard page

## M6: Use-case modules
- [x] Reindex-free migration demo; topic analytics HTML report; cross-lingual dedupe/search
- [x] Semantic cache; prefix router; document fingerprint comparison

## M7: Integrations
- [x] LangChain, LlamaIndex; PostgreSQL SQL helpers + docker-compose; Qdrant/Chroma example

## M8: Meaning Atlas
- [x] Vite + TS static app, zoomable map, in-browser encoding, fly-to, share URL, compare, accuracy radius
- [x] Deploy docs for GitHub Pages and HF Spaces

## M9: Open data
- [x] Tag multilingual Wikipedia sample; dataset card; upload script (not run)

## M10: Docs + launch kit
- [x] README, mkdocs site, Colab notebook, examples in CI, HF Space app, paper draft, launch/

## M11: Hardening
- [x] Performance, OS notes, security review, license audit, Atlas accessibility, 1.0.0rc1

## M12: Final audit
- [x] Clean clone + venv, README walk-through, examples, tests, CI, Atlas screenshot, docs vs numbers
