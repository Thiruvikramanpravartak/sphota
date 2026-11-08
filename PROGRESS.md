# Progress — final summary (v1.0.0rc1, space s1)

## Where we are (2026-11-06, all sessions merged into `main`)
- **Core build session** (this file's author): v1 s1 space, SDK/CLI, adapters, benchmark, D-017 gate
  experiments, v1.1 regions layer, repositioning. **Paper session** (PR #2, merged): paper rewritten
  as a negative-and-positive result; launch drafts on hold. **Atlas session** (PR #3, merged): region
  layer, guided demo, mobile layout. **Local session** (`ccfbbdd`, D-019): Atlas coloured by the 32
  product regions, much faster rendering.
- **Region names** (2026-11-06): the 32 regions are now named by Qwen2.5-1.5B from their top member
  topics and translated into 10 languages (`c.region_label_in(lang)`), filtered by the label gate;
  R24 renamed by hand ("Law, Culture and Travel"). Assignment is unchanged.
- **CI:** fixed (repo made public; Actions minutes were the cause). All 11 jobs green on dispatch.
- **GPU run: done, FAIL (D-022).** Kaggle T4, medium scale, 3 anchors × 256/1024/4096 regions. Best:
  bge-m3 @256 passes 3/4 (coverage 67.2%, translation 88.0%, collapse 0.1%) but cross-model is 50.2%
  (bar 60%). The v1.1 positioning is now final; no `s2`.
- **Atlas "Analyze a file"** (2026-11-06): drop a CSV/TSV/TXT/JSONL (≤ 3,000 rows), encoded in the
  browser, grouped by theme (first code block) or region, CSV download, bundled multilingual sample.
  Themes beat regions on single-domain data (83% vs 29% purity, D-020).
- **Indian languages** (2026-11-06, D-021): 19 languages measured on FLORES; 13 work well
  (87–92% confident region agreement, 99%+ retrieval), 4 partial, Manipuri/Santali fail. Region names
  added in te, kn, ml, gu, or, as, ur (Atlas selector too); review sheet `reports/region_names_review_more.csv`.
- **Atlas upgrade** (D-023): transformers.js 4, Vite 8, TypeScript 7; new "exact codes" switch gives
  browser codes identical to Python (48/48); fixed a negative-radius canvas error on new markers.
- **Early-user readiness** (2026-11-07): fresh install from GitHub verified (~25 s + one-time model
  download). Cell labels hidden by default for single texts (D-024); README install points at GitHub
  until the PyPI release, with the download size and "analyze a file" as the first thing to try.
- **Adapter study (D-028):** nonlinear adapters double cross-model agreement at 4096 cells
  (16.0% → 34.1%) and reach 56.1% at 256 cells (gate bar 60%); still improving with data.
- **Security (D-026, D-027):** pinned + hash-verified models, pinned CI actions, isolated release
  with wheel-content check and attestations, CodeQL/Scorecard/zizmor/pip-audit/npm audit, Atlas
  without third-party code.
- **v1.1 adapters, now the default (D-029, D-030):** English same-cell agreement with the
  anchor roughly doubles for all five models; certification and migration recall unchanged.
- **Six-domain evaluation (D-032):** MASSIVE, Banking77, Stack Exchange, arXiv, IndicReviews,
  CodeSearchNet. Themes competitive on science/technical/reviews; short commands agree across
  languages only 36–56% (region); code and its description share a region 77–96%.
- **Paper rewritten** (`docs/paper/sphota.md`, PDF + arXiv LaTeX in `docs/paper/build/`), tables
  generated from `bench/results` (`scripts/paper_tables.py`, checked in CI). Publication intent D-031.
- **Next (unchanged plan):** fix Actions billing → native-speaker label review (regions first: only
  32 names) → GPU run (`PROGRESS.md` command) → decide on publishing the Atlas.


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

## Owner review follow-ups (end of session)
- **Not for launch** on the current pitch (D-015). Launch kit marked on hold.
- **Confidence coverage:** 26.0% of held-out sentences get L1 confidence ≥ 0.7 (27.9% excluding
  Portuguese; 81% on training data). A confident English code is matched by its translation 66.7%
  of the time; when both codes are confident (16% of pairs), 88.7%.
- **Portuguese anomaly (D-016):** not a data/encoding bug; 52.8% of pt test sentences land in
  Portuguese-only cells created by a homogeneous pt Wikipedia sample. Needs a rebuild → GPU run.
- **Labels:** automatic filter added (`sphota.build.label_filter`); 69,642 / 71,440 kept, 27 sensitive
  cells held. Review queue saved in `reports/labels_review.json.gz`. Native review still needed.
- **CI:** last failures were formatting only; fixed.

## CPU experiments against the D-017 gate (final session)
| build | L1 cells | coverage ≥ 60% | translation (confident) ≥ 80% | cross-model ≥ 60% | collapse ≤ 5% |
|---|---|---|---|---|---|
| v1 (frozen) | 4096 | 29.4% FAIL | 65.1% FAIL | 15.7% FAIL | 33.8% FAIL |
| v1 re-clustered | 64 | 65.0% PASS | 78.3% FAIL | 57.1% FAIL | 0.0% PASS |
| v1 re-clustered | 32 | 69.0% PASS | 79.3% FAIL | 64.5% PASS | 0.0% PASS |
| v2-mini (diversified sample) | 32 | 63.9% PASS | 78.8% FAIL | 62.3% PASS | 0.1% PASS |
| v2-mini | 4096 | 30.9% FAIL | 65.0% FAIL | 16.3% FAIL | 26.8% FAIL |
| LaBSE anchor (20k corpus) | 32 | 66.5% PASS | 88.7% PASS | 53.0% FAIL* | 0.0% PASS |

\* reduced cross-model panel (other anchor + MiniLM-L6). No configuration passes all four, and every
near-pass uses only 32–64 top-level cells, which D-018 rules out as evidence for "GPS for meaning"
(amended bar: ≥ 256 cells). Files: `bench/results/l1-sweep.json`, `anchor-check.json`,
`v2-mini-gate.json`, `v2-mini-4096-gate.json`; experimental build `experiments/v2-mini/`.

## Recommendation (superseded: the GPU run was done and failed, D-022)
- **A GPU run is worth one bounded attempt (≈ US$5–10, 2–4 h on an L4/A10)**, because the anchor
  matters: LaBSE already beats e5-small on coverage (+11 pt) and confident translation (+4.5 pt) at
  the same granularity, and bge-m3 (the strongest multilingual model, 65% cross-lingual L1 agreement
  through its adapter) was never tested as an anchor. Run it with the amended bar (≥ 256 cells).
- **Expectation: it will probably not pass at ≥ 256 cells.** On CPU every configuration loses
  ~10–25 pt of coverage and agreement each time the top level gets 4× finer.
- **Pivot positioning now** (independent of the GPU result): *multilingual topic analytics, routing and
  caching with readable codes, plus model migration through stored canonical vectors* (96.5–99.1%
  recall@10 across models). Use a coarse top level (32–64 regions) where agreement is 79–89%, keep
  finer levels as optional detail with explicit confidence.

## Deciding CPU experiment: LaBSE anchor at ≥ 256 regions (`bench/results/labse-fine.json`)
| L1 cells | coverage ≥ 60% | translation (confident) ≥ 80% | cross-model ≥ 60% | collapse ≤ 5% |
|---|---|---|---|---|
| 256 | 64.0% PASS | 77.2% FAIL | 26.1% FAIL | 0.0% PASS |
| 1024 | 79.6% PASS | 79.1% FAIL (−0.9 pt) | 50.7% FAIL | 0.0% PASS |

(cross-model panel: e5-small + all-MiniLM-L6-v2 mapped into the LaBSE space; 20k-text corpus.)

## v1.1 shipped (pivot)
- `c.region` / `c.region_label` / `c.region_confidence`: 32 cross-language regions
  (`src/sphota/data/regions`, separate from frozen v1); shown by `sphota encode`; `sphota analyze`
  groups by region by default (`--by l1` for the fine view).
- README/docs repositioned: "Readable, multilingual meaning codes for topic analytics, routing,
  caching and model migration"; universal fine-grained GPS moved to Roadmap/research.
- CI cost cut: push/PR = lint + Linux/3.12 fast tests; full matrix on tags / manual dispatch.

## Final recommendation (before the GPU run; outcome in D-022)
**GPU run: yes, one bounded attempt, with LaBSE and bge-m3 anchors at 1024 regions.** A strong
aligned anchor already passes coverage and collapse at ≥ 256 regions on CPU-scale data and misses
translation agreement by under 1 point at 1024. Cross-model agreement (51%) is the real obstacle: the
GPU run must improve adapters too (more paired texts, nonlinear adapters), not only the anchor. If
cross-model stays < 60%, keep the v1.1 positioning permanently (regions + stored vectors) and treat
fine codes as within-model precision.

## GPU run (exact command)
```bash
bash scripts/gpu_build_v2.sh --scale full --anchors "sentence-transformers/LaBSE BAAI/bge-m3" --sizes "1024 4096"
```
Guide, VRAM/time estimates and renting instructions: `docs/gpu-run.md`. Dry run verified on CPU:
`bash scripts/gpu_build_v2.sh --scale tiny --dry-run` (~3 min).

## Exact next steps
1. **GPU `full` scale (space s2 candidate)** — pass/fail bar in D-015, plus the language-collapse
   gate in D-016 (diverse per-language Wikipedia sampling): `python -m sphota.build.data --scale full`,
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
