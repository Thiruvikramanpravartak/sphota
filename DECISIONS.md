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

## D-022 GPU gate, interim result (medium scale, Kaggle T4)
Interim numbers from the owner's Kaggle run (`bench/results/kaggle-medium-interim.json`):
e5-small fails (cross-model 16.2%, collapse 34.2%). LaBSE at 256 regions passes three of four bars
(coverage 69.4%, confident translation 83.5%, collapse 0.2%) and fails cross-model at 40.1%; at
1024 regions coverage drops to 55.2% and cross-model to 29.4%. More data fixed translation (77.2%
→ 83.5% at 256 vs the CPU run) but not cross-model agreement, as D-018 predicted. Unless bge-m3
(still running) clears 60% cross-model at ≥ 256 regions, the gate fails and the v1.1 positioning
(regions + themes + stored-vector migration) is permanent for v1. No change to the frozen space.
* D-020 follow-up (2026-11-06): `sphota analyze` now defaults to `--by theme` too (same grouping as
  the Atlas). `build_report(by=...)` keeps `region` as its default for API stability. With
  `--min-count > 1`, theme names fall back to the automatic labels so no raw text is shown.
