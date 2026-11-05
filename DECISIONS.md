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
| bge-m3 | see D-012 | | 8–15 | 2.2 GB | MIT |

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
