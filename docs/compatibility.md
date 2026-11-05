# Compatibility, performance and hardening

## Platforms

| | Linux | macOS (x86-64, Apple silicon) | Windows |
|---|---|---|---|
| core (`pip install sphota`) | ✓ CI | ✓ CI | ✓ CI |
| ONNX encoder (`[encode]`) | ✓ CI | ✓ (onnxruntime wheels) | ✓ (onnxruntime wheels) |
| CLI (`[cli]`) | ✓ CI | ✓ CI | ✓ CI |
| torch adapters (`[torch]`) | ✓ | ✓ | ✓ |

Python 3.10–3.13 (CI matrix: 3.10 and 3.13 on all three OSes). All file I/O is explicit
UTF-8 (Windows defaults to cp1252). The Atlas runs in current Chrome, Edge, Firefox and Safari
(WebAssembly; WebGPU not required).

## Performance (4 vCPU container)

| operation | throughput |
|---|---|
| encode, ONNX fp32 (reference) | ~160 sentences/s |
| encode, ONNX int8 (`SPHOTA_ENCODER=int8`) | ~310 sentences/s (not bit-identical: 85% L1 agreement with fp32) |
| quantize precomputed vectors (numpy) | ~10,000 vectors/s |
| parse / similarity / neighbours | microseconds, no model |

Tips: use `encode_batch` (length-sorted batches, LRU cache of 4096 texts); set
`OMP_NUM_THREADS` on shared machines; prefer `from_vector` if you already have anchor vectors.

## Security review (M11)

* **No code execution from data.** Release/adapters are `.npz` with `allow_pickle=False` or JSON;
  adapter loading refuses other suffixes; registry adapters are SHA-256 verified; every release
  file is SHA-256 verified on load (`ReleaseError` on mismatch).
* **No `trust_remote_code`.** Models are plain ONNX graphs or sentence-transformers models.
* **Network.** Only the Hugging Face Hub, only when a model is not cached. `SPHOTA_OFFLINE=1`
  forbids it; `sphota bundle` + `SPHOTA_BUNDLE` for air-gapped use (bundle manifest with SHA-256).
* **Atlas.** Static site; text never leaves the browser; only model files are fetched from the Hub.
  User text is HTML-escaped before rendering; the share URL contains only the code.
* **SQL helpers.** Pure SQL, `IMMUTABLE`/`STABLE`, no dynamic SQL.
* **Supply chain.** Pinned JS dependencies (`package-lock.json`); `scripts/license_audit.py`.

## Accessibility (Atlas)

Keyboard: `/` focuses search, arrows pan, `+`/`-` zoom, `0` resets; skip link; all controls are
native buttons/inputs with labels; results are announced via an `aria-live` region; the panel
content is plain DOM text (screen-reader readable); `prefers-reduced-motion` disables fly-to
and pulsing; text contrast ≥ 4.5:1 on the dark theme; responsive layout down to 360 px.
