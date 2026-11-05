# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow SemVer.
The *space version* (`s1`) is independent of the library version and is frozen.

## [Unreleased]

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
