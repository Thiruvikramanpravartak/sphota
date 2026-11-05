# Progress — final summary (v1.0.0rc1, space s1)

## What was built
- **Standard:** RFC-style spec (`docs/spec/`, parts 1–3, 6–10), frozen space `s1`
  (`src/sphota/data/v1`, SHA-256 manifest), golden tests pinning every code, conformance
  procedure with certificates, governance/RFC process. The release build is deterministic: a
  rebuild reproduced every golden code bit-for-bit.
- **Library:** numpy-only core (parse/compare/distance/labels), ONNX encoder (bit-exact with
  PyTorch), `encode`/`encode_batch`/`from_vector`/`explain`/`similarity`/`neighbors`/`path`/
  alternates/top-k/`fingerprint`/`compare`/`diff`, precision truncation, stable mode, privacy by
  depth, offline bundles, migration tooling, CLI mirroring everything.
- **Adapters:** 5 certified (all-MiniLM-L6-v2, bge-small-en-v1.5, gte-small: certified;
  paraphrase-multilingual-MiniLM-L12-v2, LaBSE: certified-multilingual), ONNX paths where
  available, per-adapter calibrated confidence; `train-adapter`, `certify`; OpenAI/Cohere recipes.
- **Benchmark + leaderboard:** `bench/results/*.json`, `docs/benchmark.md`, `bench/leaderboard/`.
- **Use cases:** CodeIndex (never re-index), analytics HTML report, cross-lingual search/dedupe,
  semantic cache, prefix router, document comparison. **Integrations:** Postgres (tested on PG16),
  LangChain, LlamaIndex, Qdrant.
- **Meaning Atlas:** static Vite/TS app; in-browser encoding (codes identical to Python fp32 in
  Playwright tests), fly-to, share URLs, compare mode, accuracy radius, 11 label languages.
- **Open data:** 68k tagged Wikipedia sentences pipeline + dataset card + upload script.
- **Docs & launch:** README, mkdocs site (strict build in CI), Colab notebook, HF Space app, paper
  draft, launch kit (HN, Reddit, X, LinkedIn, video script, one-pager, 30-day plan).
- **Hardening:** CI on Linux/macOS/Windows × Python 3.10/3.13, encode job with slow tests and all
  examples, Atlas build, docs build; license audit; security/a11y notes; publish script (not run).

## Key measured results (held-out FLORES-200 devtest, 12 languages; STS-B)
| | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| translation shares prefix | 46.1% | 22.9% | 11.0% | 4.8% |
| paraphrase shares prefix | 71.6% | 53.8% | 38.3% | 27.8% |
| both codes confident ≥ 0.7 | 88.7% (16% of pairs) | 85.6% (2%) | | |
- Alignment ablation: PCA-only L1 34.1% → meaning-LDA 46.1%.
- Never re-index (index with model X, query with anchor) via canonical vectors: recall@10
  96.5–99.1% vs 100% re-embedding; codes alone 10–28%.
- Utilization 92% of 4096 L1 cells; speed 160 sent/s (fp32 ONNX, 4 vCPU).
- Negative results: cross-model whitening (D-012), vec2vec at CPU budget (0% top-1).

## Known limitations
- `small`-scale build (≈78k sentences); small anchor; very short sentences get low confidence.
- Exact code identity across models is modest (6–23% at L1); certification is neighbourhood-level.
- Labels are machine-generated; hi/ta/bn translation quality mixed; no Japanese labels.
- bge-m3 adapter not certified at the CPU budget; bge-m3 anchor bake-off not completed.

## Exact next steps
1. **GPU `full` scale (space s2 candidate):** `python -m sphota.build.data --scale full`,
   re-run the bake-off including bge-m3 as anchor, `python -m sphota.build.release --scale full
   --out <dir>`, adapters, bench; produce `s1→s2` table with `sphota migrate --build`; open RFC.
2. **Labels:** human review drive (Hindi, Tamil, Bengali); better Japanese MT; regenerate with
   `python -m sphota.build.label_llm` (labels can change without an RFC).
3. **Publishing (owner decision):** `bash scripts/publish.sh all` (PyPI, GitHub Pages, HF Space,
   HF dataset) — see RELEASING.md.
4. **Launch order:** publish → Atlas live → Show HN (launch/hacker-news.md) → X/LinkedIn same day →
   Reddit next day → follow launch/growth-30-days.md.

## Incidents during the build
- Container restart (all jobs resumable; nothing lost).
- `.gitignore` patterns excluded `src/sphota/data`/`build` for a while; fixed.
- Windows checksum failures from CRLF conversion; fixed with `.gitattributes`.
