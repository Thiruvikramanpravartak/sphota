# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow SemVer.
The *space version* (`s1`) is independent of the library version and is frozen.

## [Unreleased]
### Added
- 32 cross-language regions (`c.region`, `c.region_label_in(lang)`), names in 18 languages,
  checked by script, offensiveness and back-translation filters.
- `sphota analyze --by theme` (default): groups named by the file's own most confident texts.
- Meaning Atlas: "Analyze a file" (CSV/TSV/TXT/JSONL, in the browser), CSV export, "exact codes"
  switch (fp32 model, identical to Python), 7 more label languages.
- `sphota.unsupported_script()` and a warning when encoding Ol Chiki / Meetei Mayek text.
- `Sphota.confident_label()`; `label_confident` in `to_dict()`.
- Benchmarks: 19 Indian languages (FLORES); GPU decision gate at medium scale (failed, D-022).
- GitHub workflows for PyPI trusted publishing and GitHub Pages (manual).
- Adapter set **v1.1**, the default (`adapters="v1"` / `SPHOTA_ADAPTERS=v1` for the linear set):
  nonlinear adapters for all five registry models, about twice the same-cell agreement with the
  anchor (D-028, D-030); new adapter kind `mlp2` in spec part 7.
- Supply-chain hardening: pinned, hash-verified model downloads; pinned CI actions; isolated,
  attested releases with a wheel-content check; Atlas serves all code itself under a strict CSP.
### Changed
- Positioning: "readable, multilingual meaning codes for topic analytics, routing, caching and
  model migration" (no longer "GPS for meaning").
- `sphota encode` no longer prints cell topic labels unless `--labels` is given (D-024).
- Atlas: transformers.js 4, Vite 8, TypeScript 7.
### Fixed
- Atlas: negative canvas radius on freshly placed markers.

## [1.0.0rc1]
### Added
- Space v1 (`s1`): frozen canonical projection (meaning-LDA on e5-small), 4-level residual
  codebook (4096/256/256/256), calibrated per-level confidence, labels in 11 languages,
  native keyphrases, L1 neighbour table.
- Certified adapters: all-MiniLM-L6-v2, bge-small-en-v1.5, gte-small, paraphrase-multilingual-
  MiniLM-L12-v2, LaBSE.
- Python SDK and CLI: encode, explain, similarity, neighbors, path, alternates, top-k codes,
  fingerprint/compare/diff, from_vector with adapters, offline bundles.
- Adapters and registry; `train-adapter` and `certify`.
- Benchmark suite and leaderboard; use-case modules; integrations; Meaning Atlas.
