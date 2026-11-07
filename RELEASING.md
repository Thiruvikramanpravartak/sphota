# Releasing

Nothing is published automatically; each step below is started by the owner.

## Python package → PyPI (no token needed)

1. One time, on pypi.org → Account → Publishing → **Add a pending publisher**:
   project `sphota`, owner `Thiruvikramanpravartak`, repository `sphota`, workflow
   `release.yml`, environment `pypi`.
2. Set the version in `src/sphota/__init__.py` (e.g. `1.0.0`), update `CHANGELOG.md`, commit.
3. Tag and push: `git tag v1.0.0 && git push origin v1.0.0`. The **Release to PyPI** workflow
   checks that the tag matches the version, runs the tests, builds and publishes.

## Meaning Atlas + docs → GitHub Pages

1. One time: repository Settings → Pages → Source: **GitHub Actions**.
2. Actions → **Publish site** → Run workflow. The Atlas is served at
   `https://thiruvikramanpravartak.github.io/sphota/` and the docs at `.../sphota/docs/`.

## Hugging Face (optional)

```bash
huggingface-cli login
HF_SPACE=<user>/sphota-atlas bash scripts/publish.sh atlas-hf
HF_DATASET=<user>/sphota-wiki-s1 bash scripts/publish.sh dataset
```

## Checklist

CI green on all OSes (Actions → CI → Run workflow); `pytest -m slow` green; `CHANGELOG.md`
updated. Space versions are independent of library versions: never rebuild
`src/sphota/data/v1` for a library release (golden tests enforce this).
