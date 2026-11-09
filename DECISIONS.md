# Decision log

Each entry: decision, alternatives, why. Numbers refer to files in `bench/results/`.

## D-001 Branch
**Decision:** develop and push on `claude/eloquent-johnson-kla3lt` (the session's designated
branch) instead of `build/v1` named in the brief; one draft PR (#1) tracks the work.
**Why:** the execution environment only permits pushing to the designated branch.
**Alternative:** `build/v1`; can be created from this branch with one command
(`git push origin claude/eloquent-johnson-kla3lt:build/v1`).

## D-002 Data sources
**Decision:** FLORES-200 via `mteb/flores` (ungated parquet mirror, CC-BY-SA-4.0), SIB-200
topic labels, Tatoeba via `mteb/tatoeba-bitext-mining` (CC-BY-2.0), STS-B paraphrase pairs,
Wikipedia 20231101 (random row groups of random shards, ≤3 sentences per article).
**Rejected:** Samanantar (CC-BY-NC-4.0 on the Hub), PAWS-X (custom terms), OPUS-100 (license
unknown), `facebook/flores` (gated script dataset).

## D-003 Anchor model: `intfloat/multilingual-e5-small`
**Bake-off (tiny scale, 6 languages, K=256/64/64/64, D=128, FLORES devtest en↔x agreement):**

| Anchor | L1 agree (aligned, rkm) | L2 | CPU sent/s | ONNX size | License |
|---|---|---|---|---|---|
| multilingual-e5-small | 0.617 (0.639 at D=384) | 0.355 | 160 | 470 MB fp32 / 118 MB int8 | MIT |
| paraphrase-multilingual-MiniLM-L12-v2 | 0.619 | 0.386 | ~170 | 470 MB | Apache-2.0 |
| LaBSE | 0.666 | 0.435 | ~45 | 1.9 GB | Apache-2.0 |
| bge-m3 | not completed (encoding the tiny set at 8–15 sent/s exceeded the CPU budget) | | 8–15 | 2.2 GB | MIT |

**Decision:** e5-small. It is within a few points of LaBSE on cross-lingual agreement, ~4×
faster on CPU, has an official fp32 ONNX export (bit-exact with PyTorch: max abs diff 7e-8)
and an int8 export, and runs in the browser via transformers.js; that makes the reference
encoder identical in Python and in the Meaning Atlas, with no adapter error for the default
path. It is also stronger on topical structure than LaBSE (SIB-200 topic NMI 0.50 vs 0.48)
and paraphrase-MiniLM (0.43).
**Alternatives:** LaBSE (best alignment but 4× slower and 1.9 GB, which breaks the 60-second
quick start and the browser demo); bge-m3 (strongest model overall but 568M params, ~10
sentences/s on 4 CPU cores). Both remain available as adapters and are the natural anchor
for a GPU `full`-scale s2.

## D-004 Canonical space: PCA + meaning-LDA alignment, D = 256
**Measured (small scale, 12 languages, K1=4096):** L1 cross-lingual agreement 0.341 with
PCA-whitening vs 0.462 with meaning-LDA (+35% relative); L2 0.083 → 0.222. D = 384 gives
no gain over 256 at K1=1024 (0.549 vs 0.555), so D = 256 (smaller artifacts).
Tiny-scale sweeps: Wiener-style shrinkage of noisy directions and L2-renormalizing after
projection both hurt; lower regularization (0.01–0.05) is better than 0.1–0.5.
**Alternatives:** orthogonal Procrustes to an English pivot (needs a pivot language, does
not remove language-identity directions), contrastive fine-tuning of a projection head
(overfits ~800 meanings × 12 languages on CPU; revisit at full scale), language-mean
centering (needs language ID at encode time).

## D-005 Quantizer: residual k-means, sizes 4096/256/256/256
**Bake-off (tiny):** residual k-means (rkm) vs RQ with EMA + dead-code revival (rqema) vs
hierarchical k-means tree. rqema shows slightly higher raw agreement *only because it uses
fewer codes*: L1 perplexity 0.19–0.23 of K vs 0.29–0.37 for rkm (classic partial collapse even
with revival). The tree's deeper levels degenerate at our data scale (children of a cell with
~25 points), inflating L3/L4 "agreement" because most cells have a single child.
rkm: no collapse (small scale: 92% of 4096 L1 codes used on held-out Wikipedia, perplexity
0.53·K), compact codebooks (4096 + 3×256 centroids), and a simple frozen assignment rule.
**K1:** 4096 vs 1024 at small scale: L1 agreement 0.46 vs 0.55. We keep 4096 because L1
should name a *specific* topic area (the "corporate leadership" granularity), and precision
truncation + calibrated confidence handle uncertainty. Coarser grouping can always be added
on top (merging neighbouring L1 cells with the shipped neighbour table), but L1 cells cannot
be split later without a new space version.

## D-006 Code format
Crockford base32, widths 4/2/2/2 (20/10/10/10 bits), per-level affine permutation of indices
(`a·i + b mod 2^bits`). Makes codes look varied, and the sparse valid set rejects 85% of
single-symbol typos without a check symbol. Optional `s1:` version tag and `.X` mod-37
check symbol (depth folded in). **Alternative:** hex (4096 = 3 hex chars, 256 = 2), shorter
but no typo detection and conflicts with the requested base32.

## D-007 Confidence model
Softmax over the top-16 squared distances with a per-level temperature fitted by maximum
likelihood to predict a *held-out translation's* symbol (proper scoring rule ⇒ calibrated).
Cumulative confidence is the product across levels. **Alternatives:** raw margin
(uncalibrated), isotonic regression on margins (needs more held-out data), Monte-Carlo
noise (slower; used only for stable mode).

## D-008 Stable mode
Deterministic majority vote of nearest centroid under 32 fixed Gaussian perturbations of
scale σ (calibrated on held-out pairs), restricted to the 16 nearest centroids. Optional,
because codes in stable mode differ from default codes for near-boundary inputs.

## D-009 Labels: per-language c-TF-IDF keyphrases
Labels in each language are keyphrases from that language's members of the cell. No MT
model, no LLM (a 0.5B instruct model on CPU would take hours for 4096 + L2 cells and adds a
hallucination risk; recorded as a `full`-scale improvement).

## D-010 Packaging
Release artifacts ship *inside the wheel* (`src/sphota/data/v1`, a few MB) so parse /
distance / labels work offline with numpy only; only the encoder model downloads (from the
Hub, cached). Default encoder: fp32 ONNX of the anchor (reference); `SPHOTA_ENCODER=int8`
for a 4× smaller download.

## D-011 Adapter format
`.npz` with `A`, `b`, JSON `meta` (no pickles), checksummed in the registry. Linear ridge
maps by default (paired texts), optional single residual layer.

## D-012 Cross-model noise in the canonical space: tried, not adopted (negative result)
**Problem:** a linear adapter for all-MiniLM-L6-v2 agreed with the anchor at L1 on only 16.6% of
English test sentences (cosine fidelity 0.55; a 1-hidden-layer MLP reached 24.7%).
**Hypothesis:** the space is noise-whitened against e5's *own* translation scatter, so it
amplifies directions in which e5 is self-consistent but which other models do not encode.
**Experiment:** add the scatter of cross-fitted residuals of linear maps (reference model →
anchor) to `S_within` with weight λ (`fit_space(model_residuals=…)`), holding out bge-small.
At K1=1024 (scratch sweep): λ=1 raised the reference model's L1 agreement 0.196→0.274 and the
held-out model's 0.182→0.214 with unchanged cross-lingual agreement; λ≥3 traded cross-lingual
agreement away (0.533→0.317 at λ=10).
**At the real size (K1=4096, 3 reference models, λ=1):** no gain (MiniLM 0.167 vs 0.166,
bge-small 0.136) and cross-lingual L1 fell 0.462→0.450.
**Decision:** keep λ=0 (the original s1). The rebuild with λ=0 reproduced every golden code
bit-for-bit (reproducibility check). The machinery stays in `build/fit.py` for full-scale
work. Consequence: exact-code agreement across models is modest in s1; certification and the
migration use case are therefore defined on recall-oriented measures (top-3 L1 candidates,
L1 neighbourhood) plus exact agreement, and each adapter carries its own calibrated
confidence temperature so `min_confidence` remains honest (D-013).

## D-013 Realistic conformance thresholds and per-adapter calibration
Thresholds were first set by analogy (L1 ≥ 0.60), which no small open model reaches. They are
re-set from measurements of four models (see spec Part 8): certification requires the anchor's
L1 cell to be among the model's top-3 L1 candidates for a minimum share of English test
sentences and a minimum fidelity; exact agreement is reported, not gated. Each adapter stores a
`temperature_scale` fitted by maximum likelihood on held-out pairs, so confidence for adapted
models is wider (more truncation) instead of over-confident.

## D-014 Labels: LLM names + MarianMT, quality-gated per language
English names come from Qwen2.5-1.5B-Instruct (L1, 4096 cells) and Qwen2.5-0.5B-Instruct (L2,
4186 populated cells; 400 non-English outputs discarded). Translations via opus-mt. Manual spot
check: es/fr/de/pt/ru/zh good; hi/ta/bn mixed (some transliteration noise) — shipped, flagged as
machine-translated, with native c-TF-IDF keyphrases kept in `keywords.json`; **ja dropped**
(opus-mt-en-jap is Bible-domain and produced nonsense), Japanese falls back to English.
L2 labels are translated into hi, ta, bn, es, fr, zh only (CPU budget). Labels are informative
data and can be improved without an RFC (a human review drive is on the roadmap).

## D-015 Owner review: do not launch on the current pitch; label quality gate
Owner review concluded the build is a strong topic-code/analytics tool but does not yet support
"GPS for meaning" claims. Measured follow-up: only 26% of held-out sentences get L1 confidence
≥ 0.7 (81% in-sample), and a confident English code is matched by its translation 67% of the
time. Plan: (1) no publishing; (2) one decisive GPU experiment (full scale, bge-m3 anchor, more
parallel data, stronger alignment) with the bar set in advance (L1 translation agreement ≥ 80%
and cross-model ≥ 60% on the confident portion, plus adequate coverage); (3) if it fails,
position as multilingual topic analytics + routing/caching + vector-based migration.
Labels now pass `sphota.build.label_filter` (offensive/sensitive cells held for review,
wrong-script and degenerate labels removed; 69,642 of 71,440 kept). Native review still required.

## D-016 Portuguese coverage anomaly: language-identity cells (not a data bug)
Investigated the 4.5% (pt) vs 21–36% (others) share of held-out sentences with L1 confidence
≥ 0.7. Data, language tags (`por_Latn`), alignment (en→pt nearest neighbour 99.2%) and encoding
are correct. Cause: **52.8% of Portuguese test sentences fall into Portuguese-dominated L1 cells**
(80–96% Portuguese Wikipedia members; 5 cells hold 53% of pt test sentences) regardless of topic;
for every other language this share is ≤ 0.2% (`bench/results/language-collapse-small.json`).
The pt Wikipedia sample is unusually homogeneous (mean pairwise cosine 0.84 vs ~0.78; dominated
by "Campeonato"/list stubs), so k-means formed language cells and pt points sit between two of
them (median top-2 gap 11.9 vs 23–31). **Not fixed in s1**: fixing it means rebuilding the frozen
space and its labels; it goes into the full-scale run, which must (a) sample Wikipedia with
topic/length diversity per language, (b) report the language-collapse metric, and (c) treat any
language with > 5% of test sentences in own-language cells as a gate failure. Coverage excluding
Portuguese: 27.9% at τ = 0.7 (26.0% including it).

## D-017 Space v2 decision gate (fixed before any v2 run)
All four criteria are measured on **held-out** sentences (FLORES-200 devtest, 12 languages; never
used to fit the space, codebooks, calibration, labels or adapters). A v2 space may be launched as
"GPS for meaning" only if it passes **all four**; otherwise the product is positioned as
multilingual topic analytics + routing/caching + vector-based model migration.

| # | Criterion | Definition | Bar |
|---|---|---|---|
| 1 | Coverage | share of held-out sentences (all 12 languages) with top-level (L1) confidence ≥ 0.7 | ≥ 60% |
| 2 | Translation agreement | among en↔x pairs where the English code has L1 confidence ≥ 0.7, share with the same L1 code (mean over 11 languages) | ≥ 80% |
| 3 | Cross-model agreement | same English text, certified adapter vs anchor, same L1 code (mean over registry adapters) | ≥ 60% |
| 4 | Language collapse | for every language, share of its held-out sentences in L1 cells whose training members are ≥ 80% that language (English excluded) | ≤ 5% for every language |

Reference (s1, small): coverage 26.0%, translation 66.7%, cross-model 6–23% (exact), collapse 52.8%
(Portuguese) → FAIL on all four.

## D-018 CPU experiments against D-017, and a loophole in the gate
**L1 sweep** (v1 canonical vectors, re-clustered top level; `bench/results/l1-sweep.json`): stability is
bought with coarseness. 4096 cells: 29% coverage, 65% confident translation agreement, 16%
cross-model, 34% worst collapse (fails all four). 32 cells: 69% / 79.3% / 64.5% / 0% (passes 3 of
4; translation misses 80% by 0.7 pt).
**v2-mini** (diversified per-language sample: ≤ 40 sentences per title category, near-duplicates
removed, ≤ 8000 per language; same anchor; `bench/results/v2-mini-gate.json`): at 32 cells
63.9% / 78.8% / 62.3% / 0.1% (3 of 4), no better than re-clustering v1. Kept only as an
experimental build in `experiments/v2-mini/` (labels: c-TF-IDF + label filter); frozen v1 untouched.
At 4096 cells the diversified sample cuts the worst collapse only from 33.8% to 26.8% (still
Portuguese), so D-016's cause is partly the anchor itself (e5-small separates Portuguese), not just
the sample. **Anchor check** (`bench/results/anchor-check.json`, same 20k corpus): LaBSE vs e5-small
at 32 cells: coverage 66.5% vs 55.8%, confident translation 88.7% vs 84.2%, cross-model 53.0% vs
60.3%.
**Loophole:** the tiny-scale dry run of `scripts/gpu_build_v2.sh` "passed" all four criteria with
16 top-level cells and 6 languages. The gate can be passed by making the top level coarse enough,
which says nothing about "GPS for meaning". **Amendment (applies to any v2 claim):** the gate must
be met with **≥ 256 top-level cells**, on all 12 evaluation languages, with the cell count declared
before the run. Results at 32–64 cells support a *coarse topic* product, not a coordinate system.

## D-019 Atlas shows the 32 product regions, not its own 24 clusters

The Atlas used to colour the map by 24 agglomerative "continents" made only for the map, while the
product's coarse layer is the 32 regions in `sphota.regions` (R00..R31). The Atlas now colours each
L1 cell by the region of its centroid, opens on the 32-region overview, and names a typed
sentence's region in the browser (`atlas/public/data/regions.json`, mirroring `region_of`). The 2D
layout (t-SNE) is unchanged: `python -m sphota.build.atlas_data --regions-only` re-colours an
existing `map.json`. Regions are looser on the t-SNE layout than the old clusters (63% vs 75% of
5 nearest map neighbours share a group), so each region label sits at its densest cluster of cells,
not its mean position.

## D-020 "Analyze a file" groups by theme (first code block), not by region
The Atlas can now analyze a CSV/TSV/TXT/JSONL file in the browser (up to 3,000 rows). On the bundled
48-message multilingual support sample (`atlas/public/data/sample-feedback.csv`, 10 hand-tagged themes,
en/hi/ta/bn/es), grouping by the 32 regions gives 5 groups, purity 29%, with 60% of messages in one
region (R17); Tamil messages lean to R24 whatever the topic. Grouping by the first code block (L1 cell)
gives 22 groups, purity 83%, largest group 21%, and several groups hold the same complaint in 4–5
languages (payment failed: WJX1; late delivery: FN0K). Regions are fitted on general Wikipedia text, so
one domain falls into a few of them. Decision: the file report groups by theme (L1) by default, names
each theme by its most confident message from the file (the Wikipedia-derived L1 labels are misleading
out of domain), and keeps "32 regions" as a toggle. Follow-up: the same default for `sphota analyze`
on single-domain files, and merging near-duplicate themes.

## D-021 Indian-language coverage: measured, and two languages ruled out
`python -m sphota.build.indic --scale small` measures the frozen s1 space on FLORES-200 devtest
for 19 Indian languages against English (`bench/results/indic-small.json`, docs/benchmark.md).
13 languages work well (87–92% region agreement when both codes are confident, retrieval@1 ≥ 99%);
Assamese, Sindhi, Sanskrit and Kashmiri are partial (78–83%); Manipuri (Bengali script) and Santali
(Ol Chiki) fail. Santali is confidently wrong (94% "confident", 97.5% in one region), so confidence
alone does not protect users there; the docs say so explicitly. Region names were machine-translated
(OPUS-MT) into te, kn, ml, gu, or, as and ur and passed the label filter. Punjabi and Nepali were
dropped: opus-mt-en-inc ignores `>>pan<<`/`>>npi<<` and returned mixed Hindi/Bengali/Marathi
script, which the script check did not fully catch. Follow-ups done: the label filter now rejects labels
with letters from a second non-Latin script, and `encode` warns on Ol Chiki / Meetei Mayek input
(codes unchanged, frozen behaviour).
The new names go on a separate review sheet (`reports/region_names_review_more.csv`).

## D-022 GPU gate result (medium scale, Kaggle T4): FAIL. Positioning is final for v1
Full run (`bench/results/v2-gate-medium-*.json`, table in docs/benchmark.md): no anchor passes all
four D-017 bars at ≥ 256 regions. Best: bge-m3 at 256 regions, 3/4 (coverage 67.2%, confident
translation 88.0%, collapse 0.1%) with cross-model agreement 50.2% vs 60%; LaBSE at 256, 3/4 with
40.1%; e5-small fails everything (collapse 34–41% even with the diversified sample). Cross-model
agreement falls with every 4× refinement for every anchor (bge-m3: 50 → 38 → 25%). Decision, as
pre-registered: Sphota v1 stays "readable multilingual meaning codes for topic analytics, routing,
caching and model migration" (32 regions + themes + stored canonical vectors); fine codes are
within-model precision. No `s2` space is planned on this evidence; the frozen s1 space is unchanged.
What could change it: nonlinear or jointly-trained adapters (cross-model, not anchor, is the
bottleneck); out of scope for v1.

## D-023 Atlas dependencies upgraded; "exact codes" switch
transformers.js 3.7.6 → 4.3.2, Vite 6.4.5 → 8.3.6, TypeScript 5.9.3 → 7.0.2 (build and type check
clean). The browser's default model is the q8 ONNX (~120 MB). Its codes were never identical to
the Python reference: on the 48-message sample, q8 matched Python's full code 22/48 (v3) and 25/48
(v4), and the first block 44/48 vs 42/48. The fp32 ONNX in transformers.js 4 matches Python **48/48**
(`atlas/tests/sample-codes.mjs`). So the upgrade is kept and the Atlas gets an "exact codes" switch
(fp32, ~470 MB; also `?precision=fp32`); q8 stays the default for speed, and the report says which
was used. Stored codes should come from the Python library or exact mode.

## D-024 Cell topic labels are hidden by default for single texts
On everyday text the L1 cell labels (named from Wikipedia topics) mislead: "मेरा ऑर्डर अभी तक नहीं
आया" → "Kali Yuga Tenets and Authors"; "Lionel Messi scored twice in the final" → "Golf
Championships" (L1 confidence 0.79). Hiding labels below L1 confidence 0.7 shows them on only 7 of
the 48 sample messages, and those 7 are still wrong ("payment failed" → "Lottery vs Gambling"), so
confidence alone does not fix it. Decision: `sphota encode` no longer prints cell labels unless
`--labels` is passed (and then only at L1 confidence ≥ 0.7); new `Sphota.confident_label()` and a
`label_confident` field in `to_dict()`; the Atlas sentence panel shows the label only at ≥ 0.7,
marked "nearest general topic". `c.label` itself is unchanged (API stability). For people-facing
names, use themes named by the user's own texts (D-020) or regions.

## D-025 Automatic back-translation check for region names
Machine-translated region names included nonsense that the script/length filter cannot see
(Telugu "Filmography" → "హైటెక్‌గ్రౌంటీ", back-translated "[Picture on page 17]"; Bengali
"Automotive Innovations" → "Unique Violence"; Arabic "Filmography" → "أولاً-", "Firstly-").
Raw anchor-model similarity does not separate good from bad names (all 0.7–0.95), so the check is
a ranking test: among the 32 English region names, the translated name or its OPUS-MT
back-translation must point to its own region within the top 2 (`python -m
sphota.build.label_llm --step region-check`). 30 of ~550 names failed (11 languages; most in Urdu
6, Odia 4, Hindi 3) and were removed, so those regions show in English. The review sheets mark
them for native speakers. Region assignment and English names are unchanged.

## D-026 Supply-chain hardening: pinned, hash-verified models
The library downloaded models from the Hugging Face Hub at whatever revision was current. A
compromised or simply updated model repo would silently change codes (breaking the frozen
standard) or serve a malicious file. Now `sphota/data/models.json` pins every model the library
downloads to a Hub commit (e5-small at 614241f6…, the commit the s1 release was built from) with
the SHA-256 of each file; `hf_file` downloads at that commit and refuses any file whose hash
differs (also inside offline bundles). sentence-transformers models load at a pinned revision with
`trust_remote_code=False`. Verified files are remembered (path, size, mtime) so the 1.3 s hash of
the fp32 model is paid once. Unknown (user-chosen) models are not pinned.

## D-027 Release, CI and Atlas security hardening (before the first public release)
Built against the 2026 LiteLLM incidents (stolen PyPI token via a poisoned CI action; hijacked
pipeline shipping a `.pth` startup hook). CI: every action pinned to a commit SHA (versions ≥ 14
days old; CodeQL held at v4.39.1 for that reason), workflow-level `permissions: {}` or
`contents: read`, `persist-credentials: false`, no caches in jobs that run on tags (zizmor
cache-poisoning findings), zizmor + pip-audit + npm audit + a wheel-content check on every push,
CodeQL and Scorecard workflows, Dependabot with a 14-day cooldown, CODEOWNERS for CI/data. Release:
separate test / build / publish jobs; hash-pinned build tools (`uv pip compile --exclude-newer`);
`scripts/check_wheel.py` (wheel == git-tracked files, no `.pth`); trusted publishing with PEP 740
attestations behind a reviewer-gated environment. Atlas: transformers.js 4 imported the ONNX
Runtime (a dev build) from cdn.jsdelivr.net at run time; it is now served from the site
(`vite.config.ts`), enforced by a CSP that only allows the site and Hugging Face; Google Fonts
removed (the page promised nothing is sent anywhere); browser model pinned to a commit; CSV export
neutralizes spreadsheet formulas. Kaggle notebook no longer puts a GitHub token in the clone URL.
History scanned for credentials: none.

## D-028 Adapter study: nonlinear adapters double cross-model agreement
The GPU gate failed on cross-model agreement (D-022). Holding the space, texts and test set fixed
and changing only the adapter (5 models, held-out FLORES devtest): ridge (shipped) averages 16.0% /
32.6% / 65.6% same-cell agreement at 4096 cells / 256 cells / 32 regions; a residual MLP (ridge +
one 1024-unit GELU layer, MSE) averages 34.1% / 56.1% / 71.7%. Code-aware losses (cross-entropy
towards the anchor's cell at L1, or L1 + regions) did not beat MSE on average: they gain a little
at L1 for multilingual models but give up coarse agreement. Ridge does not improve with more
pairs (2.5k → 40k); the MLP keeps improving (pmMiniLM L1 27.0% → 32.8%). Conclusion: a large part
of the "models disagree" failure was the linear adapter, not the models. The 256-cell level is
now 3.9 points from the gate's 60% bar with the e5 anchor on CPU-scale data. Next: ship MLP
adapters (opt-in, without touching the frozen v1 registry), then a larger paired-data run.

## D-029 Ship the nonlinear adapters as an opt-in set (v1.1), not as a replacement
`build/adapters_v11.py` trains ridge + residual GELU(tanh) adapters for the five registry models and
certifies the shipped numpy forward pass (float16 hidden weights): English same-cell agreement
13.8–23.3% → 32.8–36.6%, all certified (two certified-multilingual), migration recall@10
unchanged (~100%), 9.7 MB total. They live in `sphota/data/adapters/v1.1/` so the frozen v1
registry is untouched (owner rule). Default stays `v1`: switching would change stored codes for
anyone who encoded through an adapter. Making v1.1 the default is an owner decision for the next
minor version.

## D-030 v1.1 adapters become the default before the first release
The owner confirmed nobody uses Sphota yet, so switching the default adapter set cannot change
anyone's stored codes. `DEFAULT_SET = "v1.1"` in `sphota.adapters`; the linear v1 set stays frozen
inside the release and selectable (`adapters="v1"`, `SPHOTA_ADAPTERS=v1`). Release stays on hold
until the owner says it is ready to roll out.

## D-031 Publication intent: code, results and paper will be published
The owner confirmed (2026-11-08) that the project is being built for publication: the open-source
code and artifacts, the benchmark results and a research paper. The rollout date stays the owner's
call ("ready to roll out"). Consequences, binding from now on: every evaluation dataset must have
a license that allows publishing results (recorded per dataset in `docs/benchmark.md`); every
number in the paper must be reproducible by a script in this repository with its result JSON in
`bench/results/`; negative results are reported, not dropped; no private or personal data enters
the repository; the paper and docs use the D-015 positioning, with "GPS for meaning" as the stated
long-term research goal.

## D-032 Six-domain evaluation: what holds outside FLORES
`python -m sphota.build.domains --scale small` (`bench/results/domains-small.json`, license-clean
data, D-031). (1) Grouping: Sphota's frozen themes (first code block) reach V-measure 0.21–0.53,
below k-means fitted on each dataset at the same number of groups (0.27–0.74); the gap is small for
science (−0.06), reviews (−0.05 to −0.07) and technical Q&A (−0.09), larger for short commands
(−0.12 to −0.18) and largest for the narrow banking domain (−0.31). Themes beat TF-IDF clustering on
technical, science and reviews by 0.25–0.29 V, tie on conversational English and lose on banking.
(2) Cross-lingual on real text is harder than FLORES: MASSIVE commands share the English region
36–56% of the time (theme 25–46%); IndicReviews sentences share the Hindi region 63–73% (Bodo,
unsupported, 47.5%). (3) Cross-model on real text: v1.1 adapters raise same-theme agreement
17.2% → 31.6% (en) and 15.1% → 28.4% (hi). (4) Code: docstring and code share a region 77–96% and a
theme 34–55%; docstring→code recall@10 75–94% with raw anchor vectors, 76–93% after the canonical
projection. Product consequence: themes are a good zero-setup, stable-ID view; for best grouping
within one file, a per-file refinement step is the next feature to build. Claims in docs must cite
the real-data numbers, not only FLORES.

## D-033 Pre-registration: per-file refinement targets and evaluation protocol (before any run)
Recorded 2026-11-09, before implementing or evaluating any refinement method. The owner's rule:
**no bar is lowered**. The D-017/D-018 gate stays exactly as defined (coverage ≥ 60%, confident
translation ≥ 80%, cross-model ≥ 60%, collapse ≤ 5%, at ≥ 256 cells).

Targets for per-file refinement (all must hold; a miss is reported as a miss):
* **R1 quality.** On the *test* split of every evaluation dataset (and every evaluated language),
  the refined grouping's V-measure ≥ reference − 0.05, where reference = max(k-means on the
  dataset's own anchor embeddings with the true number of labels; k-means with the same number of
  groups as the refined output). The method never sees labels or the true number of labels.
* **R2 stability.** Every message's main code is identical with and without refinement (100%), and
  refinement is deterministic: the same file gives the same groups; shuffling the rows gives an
  adjusted Rand index ≥ 0.95 against the unshuffled result.
* **R3 languages.** In mixed-language files built from parallel data (pivot + one other language
  in one file), the share of parallel pairs that land in the same refined group must be ≥ the
  share that land in the same fixed theme.
* **R4 speed.** 3,000 messages refined in ≤ 2 s in Python on the reference CPU and ≤ 5 s in the
  browser, encoding excluded.

Protocol:
* Each dataset is split once into dev (30%) and test (70%) by a fixed seed (20261109). Every method
  choice and hyperparameter is made on dev only; each pre-registered candidate is evaluated on test
  once, and all results are reported, pass or fail.
* Uncertainty: 95% bootstrap confidence intervals (1,000 resamples of items) for every headline
  metric; R1 is judged on the point estimate and the CI is reported next to it.
* Candidates declared now: **M1** k-means in the canonical space with k chosen by a silhouette
  sweep; **M2** threshold clustering in the canonical space, the distance threshold expressed in
  translation-noise units (the space's unit, §3.2 of the paper) and fixed once on dev data across
  all datasets; **M3** split-and-merge of the fixed themes (split a theme when its members are
  spread beyond the threshold, merge themes whose centres are within it). Variants of these may be
  tried on dev; anything not listed here is reported as exploratory.
* Datasets must be license-clean (D-031); FLORES-derived sets are excluded from evaluation of
  anything fitted on FLORES dev.

## D-034 Amendment to D-033, recorded before the small-scale test run
Disclosure: a tiny-scale smoke run (300-row subsets, used to check the pipeline) reported its test
split, which showed the unmodified M3 failing R3 (mixed-language files: refinement split language
pairs more often than the fixed themes). Diagnosis on dev data only: on MASSIVE the per-language
mean offset in the canonical space is 6–11 (0.3–0.5 translation units), against 1–2 on FLORES, so
2-means splits mixed groups along language. Changes, all chosen and checked on dev splits; the
small-scale test split has not been evaluated by any method:
* Variants added to M2/M3 (declared family): per-script mean centring (`language="centre"`) and a
  split guard that refuses splits separating scripts (`language="guard"`); script = dominant
  Unicode writing system of the text (label-free). M3_auto: split-and-merge with tau chosen per
  file by silhouette over a fixed grid (label-free).
* tau grid extended to 0.3–1.4 (the dev optimum sat at the old grid edge, 0.6).
* Selection rule: a candidate is eligible only if it passes R3 on dev mixed-language files; among
  eligible candidates, maximize the mean dev gap (V − reference), as before.
* Known risk of centring: if language and topic are correlated in a real file, centring can merge
  topics; the guard does not move vectors and has no such risk.
Targets R1–R4 and their bars are unchanged.

## D-035 Pre-registration: GPU gate run 2 (strongest combination), before it is run
The D-017/D-018 gate is unchanged. Run 2 tests the combination the evidence points to: anchors
LaBSE and bge-m3 (run 1: 3/4 at 256 cells, cross-model 40–50%), nonlinear adapters (D-028: about
2× same-cell agreement), and MASSIVE train commands (11.5k meanings aligned across the 12 gate
languages, test texts excluded) added to the meaning-LDA translation scatter, targeting the
short-text weakness (D-032). Medium scale, sizes 256 / 1024 / 4096, notebook cell 4.
Rules fixed now: (1) 6 configurations are tested (2 anchors × 3 sizes); this number is reported.
(2) A configuration passes only if all four criteria pass. (3) Because 6 tries raise the chance
of a lucky pass, a pass counts as a gate result only after the same configuration also passes a
confirmation run with a different k-means seed (`--seed 1`); otherwise it is reported as
"passed once, not confirmed". (4) Held-out MASSIVE test commands' same-cell agreement is reported
for every configuration as a secondary, non-gate metric. (5) No result changes the frozen s1
space; a confirmed pass would open an RFC for a space s2.

## D-036 Refinement round 1 result (FAIL) and pre-registration of round 2
**Round 1 (D-033/D-034), test split, as registered:** no candidate meets R1–R4.
M3_auto+centre (dev winner): R1 15/22, R2 fails (shuffle ARI 0.08: its silhouette sample depended
on row order, a bug), R3 5/8. M2_bisect τ=1.0+centre: R1 11/22, R2 pass, R3 7/8.
M3_split_merge τ=1.0+centre: R1 8/22, R2 pass, R3 7/8. R4 (Python, 3,000 rows): 1.2 s, pass.
Diagnosis: a fixed distance threshold does not transfer across languages and domains (τ=1.0 gives
8 groups for Banking77's 77 intents and 882 for Stack Exchange's 20 sites); per-file adaptivity
helped most; split-and-merge of the fixed themes inherits their boundaries, and at equal group
counts plain k-means on the file is better. Round 1's test splits are now spent for selection.

**Round 2, registered before any round-2 run:**
* Candidate M4: deterministic k-means on the file's canonical vectors (rows in canonical order,
  k-means++ with a fixed seed, so results do not depend on row order), with per-script centring
  as an option, k chosen label-free by silhouette on a deterministic sample over a fixed grid.
  The M3_auto ordering bug is fixed and M3_auto re-enters as a candidate.
* Selection: on the round-1 dev splits, same rule as D-034 (R3-dev eligible, max mean gap).
* Evaluation: **only the single winner**, once, on a fresh confirmation set never evaluated by any
  method: MASSIVE de, fr, ru, ur, kn, ml (all test items); IndicReviews gu, kn, ml, or, pa, ur;
  MasakhaNEWS ibo, lin, lug, orm, pcm, run, sna, som, tir, xho; Stack Exchange test split #2 and
  arXiv split #2 (5,000 items each); Banking77 train split (3,000 items). R3 on mixed files
  MASSIVE en+{de, fr, ru, ur} and IndicReviews hi+{gu, kn, ml}. Targets R1–R4 unchanged.

## D-037 Refinement round 2 result (FAIL on R1) and pre-registration of round 3
**Round 2 (D-036), fresh confirmation set, winner M4_kmeans+centre, evaluated once:**
R1 16/25 (FAIL), R2 pass (shuffle ARI 1.0), R3 7/7 pass (translations share a refined group
63–80% vs 29–46% for fixed themes), R4 pass (0.31 s for 3,000 rows). Banking77 (train split):
0.742 vs reference 0.741, up from 0.416 with the fixed themes. Diagnosis: the silhouette picks too
few groups (4 for most MasakhaNEWS languages, 12 for MASSIVE de with 59 intents); V-measure falls
quickly below the true group count and stays flat above it, so undershooting is the costly error.
For languages the anchor barely knows (som, tir, xho) the fixed themes beat per-file clustering.

**Round 3, registered before any round-3 run.** Candidates (M4 family, + centre): k chosen by
(a) silhouette (round 2), (b) ⌈√n⌉, (c) ⌈2√n⌉, (d) max(silhouette k, ⌈√n⌉). Selection on the
round-1 dev splits (D-034 rule). The single winner is evaluated once on fresh confirmation set 2:
MASSIVE it, pt, nl, pl, tr, vi, id, ja, ko, fa, he, th, sw (all test items); Stack Exchange test
split #3 and arXiv split #3 (5,000 each); Banking77 train, a disjoint 3,000-item sample; R3 on
MASSIVE en+{it, ja, tr, sw}. This is the third evaluation round of R1–R4; the count is reported.

## D-038 Refinement round 3 result (FAIL on R1, R3) and pre-registration of round 4
**Round 3 (D-037), fresh confirmation set 2, winner M4_kmeans_2sqrt+centre, evaluated once:**
R1 12/16 (misses it, nl, ja, ko by 0.05–0.08), R2 pass, R3 FAIL (en+it 11% vs 29% fixed themes,
en+tr 11% vs 32%, en+sw 14% vs 17%; en+ja passes 60% vs 36%), R4 pass (0.10 s). Diagnosis: with
~2√n groups, k-means separates same-script languages (script centring cannot tell English from
Italian); round 2's coarser silhouette k hid this. Root fix: centre by *language*, not script.

**Round 4, registered before any round-4 run.**
* A language identifier for the anchor: multinomial logistic regression on raw anchor vectors,
  trained on MASSIVE **train** utterances (all 51 locales, test texts excluded) plus FLORES-200
  dev; never on any evaluation split. Shipped as a checksummed `.npz` (NumPy inference).
  Centring groups = predicted language, applied only to languages with ≥ 30 rows in the file.
* Candidates (k = ⌈2√n⌉): (a) canonical space + script centring (round-3 winner);
  (b) canonical space + language centring; (c) raw anchor space + language centring;
  (d) raw anchor space + script centring. Selection on round-1 dev splits (D-034 rule).
* The single winner is evaluated once on fresh confirmation set 3: MASSIVE af, am, az, cy, da, el,
  fi, hu, hy, is, jv, ka, km, lv, mn, ms, my, nb, ro, sl, sq, sv, tl, zh-TW (test items); Stack
  Exchange split #4, arXiv split #4; Banking77 train, a third disjoint sample; R3 on MASSIVE
  en+{da, ro, sv, ms, el, ka}. Fourth round of R1–R4; the count is reported.

## D-039 Refinement round 4 result (FAIL on R3) and pre-registration of round 5
**Round 4 (D-038), fresh confirmation set 3, winner M4_kmeans_2sqrt[canonical, script], evaluated
once:** R1 **26/27** (only Stack Exchange split #4 misses, by 0.006), R2 pass, R3 FAIL 2/6
(same-script pairs: en+da 15% vs 36% fixed themes, en+ro 22% vs 32%, en+sv 30% vs 39%, en+ms
32% vs 37%; en+el and en+ka pass), R4 pass (0.08 s). Language centring was available but not
selected: the dev R3 set held only one same-script pair (en+es), so dev could not see the failure.
Methodological lesson recorded: the dev set must contain the failure mode being fixed. On spent
round-3/4 pairs (now dev data) language centring gives 56–70% vs 29–42% fixed themes.
Script-consistent language prediction added (a Devanagari text can no longer be labelled Bengali).

**Round 5, registered before any round-5 run.** k = ⌈2√n⌉. Candidates: [canonical, script],
[canonical, language], [canonical, language with script mask], [raw, language with script mask].
Dev R1 sets: round-1 dev splits. Dev R3 pairs: round-1 dev pairs plus spent MASSIVE test pairs
en+{da, ro, sv, ms, de, it, tr}. Rule: eligible = all dev R3 pairs pass; maximize mean dev gap.
The single winner is evaluated once on fresh confirmation set 4: the MASSIVE **validation** split
(never used by any step), R1 on en, de, fr, it, tr, ro, sv, hi, ta, ja, ar, ru, sw; Stack Exchange
split #5 and arXiv split #5 (5,000 each); the remaining Banking77 train items not in samples
a–c; R3 on MASSIVE validation en+{de, fr, it, tr, ro, sv, hi, ja, ar}. Fifth round of R1–R4.
