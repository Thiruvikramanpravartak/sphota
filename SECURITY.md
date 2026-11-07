# Security policy

## Reporting a vulnerability

Please do not open public issues for vulnerabilities. Use GitHub's **Report a vulnerability**
(Security tab → private security advisory) on this repository. We aim to acknowledge within
3 working days and to publish a fix and an advisory (with a CVE where appropriate) as soon as
possible after that.

## Supported versions

| Version | Supported |
|---|---|
| 1.0.x (space s1) | yes |

## What Sphota guarantees

* **No code execution from data.** Release and adapter files are `.npz` (always loaded with
  `allow_pickle=False`) or `.json`. Sphota never loads pickles, never calls `torch.load` on
  downloaded files, and loads sentence-transformers models with `trust_remote_code=False`.
* **Checksummed release.** Every file of the frozen space is listed with its SHA-256 in
  `manifest.json` and verified on load. Registry adapters carry SHA-256 checksums, verified
  before use; offline bundles are checksummed and cannot reference files outside the bundle.
* **Pinned, verified models.** Every model the library downloads is pinned to a Hugging Face
  commit and each file's SHA-256 is checked before it is loaded (`sphota/data/models.json`); a
  mismatch is refused with an error. The Meaning Atlas pins its browser model to a commit too.
* **Offline mode.** `SPHOTA_OFFLINE=1` forbids all network access (also sets `HF_HUB_OFFLINE=1`).
* **The Meaning Atlas runs no third-party code.** All JavaScript and WebAssembly (including the
  ONNX Runtime) is served from the site itself; a Content Security Policy allows network access
  only to the site and Hugging Face (model files). No fonts, analytics or CDNs. Files you analyze
  never leave the browser. CSV exports neutralize spreadsheet formulas (`=`, `+`, `-`, `@`).

## Supply chain (how releases are made)

Lessons from the 2026 LiteLLM compromises (a poisoned CI action stole a PyPI token; a hijacked
release pipeline shipped a `.pth` startup hook) are built in:

* **No long-lived tokens.** PyPI publishing uses Trusted Publishing (OIDC); the token exists only
  for the single publish job, which runs no project code. A GitHub environment (`pypi`) with a
  required reviewer gates every release.
* **Pinned CI.** Every GitHub Action is pinned to a full commit SHA; workflows default to no
  permissions; checkouts keep no credentials; CI uses no secrets. `zizmor` lints the workflows
  on every push. Dependabot proposes updates only after a 14-day cooldown.
* **Reproducible-content wheels.** The wheel is built in an isolated job with hash-pinned build
  tools (`.github/requirements/release.txt`), then `scripts/check_wheel.py` refuses it unless it
  contains exactly the git-tracked package files, byte for byte, and no `.pth` files.
* **Provenance.** Each PyPI file carries a PEP 740 attestation linking it to this repository's
  `release.yml` run at the tagged commit.
* **Continuous checks.** `pip-audit` (Python dependencies) and `npm audit` (Atlas) on every push;
  CodeQL static analysis and OpenSSF Scorecard weekly and on `main`.

### Verifying a release

```bash
pip download sphota==1.0.0 --no-deps -d dist/
pip install pypi-attestations
pypi-attestations verify pypi --repository https://github.com/Thiruvikramanpravartak/sphota dist/*.whl
```

PyPI also shows the provenance ("Verified details") on each release page.

## Privacy

Sphota codes reveal topics; they are not anonymization. See
[docs/spec/09-security-privacy.md](docs/spec/09-security-privacy.md).

## Out of scope

Models you choose yourself (custom adapters, `--model` for unpinned models) are not pinned or
verified by Sphota. Malicious input text cannot execute code, but very long inputs cost CPU time.
