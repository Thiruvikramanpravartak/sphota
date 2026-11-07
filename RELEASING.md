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

## Repository security settings (owner, one time)

* Settings → Rules → Rulesets → new branch ruleset for `main`: require a pull request (or at
  least block force pushes and deletion), require status checks `lint`, `security`, `fast`,
  and "Require review from Code Owners" (see `.github/CODEOWNERS`).
* A tag ruleset for `v*`: only you may create, update or delete version tags.
* Settings → Environments → `pypi`: required reviewer = you; deployment tags `v*` only.
* Settings → Code security: enable private vulnerability reporting, Dependabot alerts and
  security updates, secret scanning with push protection, and CodeQL (the workflow is included).
* Your account: two-factor authentication with a passkey or security key, on GitHub and PyPI.

## Checklist

CI green on all OSes (Actions → CI → Run workflow); `pytest -m slow` green; `CHANGELOG.md`
updated. Space versions are independent of library versions: never rebuild
`src/sphota/data/v1` for a library release (golden tests enforce this).
