# Security policy

## Reporting a vulnerability

Please do not open public issues for vulnerabilities. Use GitHub's
"Report a vulnerability" (private security advisory) on this repository. We aim to
acknowledge within 3 working days.

## Design guarantees

* **No code execution from data.** Release and adapter files are `.npz` (loaded with
  `allow_pickle=False`) or `.json`. Sphota never loads pickles, never uses `torch.load`
  on downloaded files, and never runs `trust_remote_code` models by default.
* **Checksums.** Every release file is listed with its SHA-256 in `manifest.json` and
  verified on load. Adapters in the registry carry SHA-256 checksums, verified before use.
* **Offline mode.** `SPHOTA_OFFLINE=1` forbids all network access (also sets
  `HF_HUB_OFFLINE=1`).

## Privacy

Sphota codes reveal topics; they are not anonymization. See
[docs/spec/09-security-privacy.md](docs/spec/09-security-privacy.md).

## Supported versions

| Version | Supported |
|---|---|
| 1.0.x (space v1) | yes |
