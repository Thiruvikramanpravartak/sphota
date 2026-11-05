# Contributing to Sphota

Thanks for helping build an open coordinate system for meaning.

## Ground rules

* **The v1 space is frozen.** Code changes must never change the code a given vector maps to.
  `tests/test_golden.py` pins golden codes; if it fails, your change breaks the standard.
  Changes to the space itself go through the RFC process in [GOVERNANCE.md](GOVERNANCE.md)
  and ship as a new version (v2) with a migration table.
* Every feature needs a test, docs, and a runnable example.
* Keep the light core (`codes`, `release`, `space`, `distance`, `labels`) numpy-only.

## Development setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
pytest                      # fast tests (no network)
pytest -m slow              # tests that download models
ruff check . && ruff format --check . && mypy
```

## Adding an adapter

1. `sphota train-adapter --model <hf-id> --out adapters/<name>.npz`
2. `sphota certify --model <hf-id> --adapter adapters/<name>.npz`
3. Add the certificate JSON to `adapters/registry.json` with a PR. Only models with
   OSI-approved or similarly open licenses are accepted into the default registry.

## Commit style

Conventional-ish: `area: short summary` (e.g. `cli: add --json to encode`).
Sign your commits off if your employer requires it (DCO is welcome, not required).
