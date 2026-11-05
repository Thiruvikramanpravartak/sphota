# Releasing

Nothing is published automatically. When the owner decides to go public:

```bash
export UV_PUBLISH_TOKEN=pypi-...            # PyPI
huggingface-cli login                       # Hugging Face
HF_SPACE=<user>/sphota-atlas HF_DATASET=<user>/sphota-wiki-s1 bash scripts/publish.sh all
```

Checklist: CI green on all OSes; `pytest -m slow` green; `CHANGELOG.md` updated; tag
`v1.0.0rc1`; GitHub release notes link the benchmark; README badges point to the public URLs.

Space versions are independent of library versions: never rebuild `src/sphota/data/v1`
for a library release (golden tests enforce this).
