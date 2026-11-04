# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow SemVer.
The *space version* (`s1`) is independent of the library version and is frozen.

## [Unreleased]

## [1.0.0rc1]
### Added
- Space v1 (`s1`): frozen canonical projection, 4-level residual codebook (4096/256/256/256),
  calibrated per-level confidence, multilingual labels, L1 neighbour table.
- Python SDK and CLI: encode, explain, similarity, neighbors, path, alternates, top-k codes,
  fingerprint/compare/diff, from_vector with adapters, offline bundles.
- Adapters and registry; `train-adapter` and `certify`.
- Benchmark suite and leaderboard; use-case modules; integrations; Meaning Atlas.
