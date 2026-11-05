# Progress

_Last updated: M10–M12 (labels being translated; final audit next)._

## Done
- **R**: licenses verified, tiny bake-off (e5-small, multilingual MiniLM, LaBSE; bge-m3 did not
  finish on CPU), gap analysis, small-scale design sweep. Decisions D-001…D-013.
- **M0** foundation; **M1** spec (parts 1–3, 6–10); **M2** s1 release (frozen, checksummed,
  rebuild reproduces every golden code bit-for-bit).
- **M3** SDK + CLI; numpy-only core; ONNX encoder (bit-exact with torch); fresh-venv install of
  the wheel verified (no torch imported); core coverage: codes 100%, space 100%, release 97%,
  distance 95%, api 90% (total incl. model-dependent modules 84%).
- **M4** 4 certified adapters (all-MiniLM-L6-v2, bge-small-en-v1.5, gte-small: certified;
  paraphrase-multilingual-MiniLM-L12-v2: certified-multilingual), ONNX encoders for adapters,
  per-adapter calibrated temperature, train-adapter/certify, OpenAI/Cohere recipes.
- **M5** benchmark JSON + docs/benchmark.md + static leaderboard.
- **M6** use cases; **M7** integrations (Postgres tested on a real PG16 cluster).
- **M8** Atlas: in-browser encoding verified with Playwright (codes identical to Python fp32).
- **M9** 68k-sentence tagged Wikipedia sample + dataset card + upload script (not run).
- **M10** README, mkdocs site (strict build in CI), notebook, HF Space app, paper draft, launch kit.
- **M11** license audit script (all Hub assets permissive), compatibility/security/a11y notes,
  Windows fixes (UTF-8 I/O, .gitattributes for checksummed files), publish script (not run).
- CI green on Linux/macOS/Windows × Python 3.10/3.13, encode job (slow tests + examples), Atlas build.

## Doing
- Label translation (opus-mt) → merge → continent names → Atlas data + screenshot.
- LaBSE / bge-m3 adapters and the vec2vec run (CPU-heavy, after labels).

## Next
- M12 final audit (clean clone, README walk-through, docs vs numbers), PR description.

## Incidents / blockers
- Container restarted once (all jobs resumable; nothing lost).
- `.gitignore` patterns `data/`/`build/` silently excluded `src/sphota/data` and
  `src/sphota/build`; fixed (anchored patterns) and committed.
- Disk reads ~12 MB/s after the restart made large-model loading slow.
- bge-m3 bake-off not completed (8–15 sentences/s on 4 vCPU).

## Measured numbers (small scale, held-out FLORES-200 devtest, 12 languages)
- Cross-lingual agreement L1/L2/L3/L4: 46.1 / 22.9 / 11.0 / 4.8% (PCA-only: 34.1 / 8.3%).
- Paraphrase agreement (STS-B ≥ 4.5): 71.6 / 53.8 / 38.3 / 27.8%.
- Stable mode: +0.6 pt L1 cross-lingual (small).
- Utilization: 92% of 4096 L1 cells, perplexity 2182, largest cell 0.29%.
- Cross-model migration recall@10 via canonical vectors: 96.5–99.1% (codes only: 10–28%).
- Adapters: exact L1 agreement with anchor 6–23%; neighbourhood agreement 62–64%.
- Speed: 160 sent/s fp32 ONNX, 313 int8, 10.8k vectors/s quantization (4 vCPU).
