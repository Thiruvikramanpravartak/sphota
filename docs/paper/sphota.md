# Can Meaning Have Universal Coordinates? A Negative-and-Positive Result

*Draft for arXiv (cs.CL; cross-list cs.IR). The Sphota Authors, 2026. Not yet submitted.*

## Abstract

We tried to build a *universal coordinate system for meaning*: a frozen, open, hierarchical
code (e.g. `JYZA-7M-X0-A8`) that any text in any language would receive, reachable from many
embedding models through linear adapters, so that translations, paraphrases and different
models agree on the same address. Before running the decisive experiments we fixed a
four-part pass/fail gate on held-out data (coverage, translation agreement, cross-model
agreement, language collapse), and later amended it to close a loophole: the gate must be
met with at least 256 top-level cells. **No configuration we built passes.** The frozen
release (4096 top-level cells, multilingual-e5-small anchor) fails all four criteria
(coverage 29%, confident translation agreement 65%, cross-model 16%, worst language collapse
34%). Agreement is bought with coarseness: each 4× finer top level costs roughly 10–25
points on the gate metrics, and every near-pass uses 32–64 cells. The anchor matters (LaBSE
beats e5-small by 11 points of coverage and 4.5 points of confident translation agreement at
the same granularity) but not enough to change the conclusion on CPU. We also document a
*language-collapse* failure: at ≥ 256 cells, k-means forms cells that are mostly one language
(up to 55% of Portuguese test sentences), which a diversified sample only partly fixes. What
does work: (1) *coarse regions* (32–64 cells) where translations agree 79–89% of the time
when confident, useful for multilingual topic analytics, routing and caching; and (2)
*migration through stored canonical vectors*: an index built with one model can be queried
with another at 96.5–99.1% recall@10 without re-embedding. We release everything, including
the gate, the negative results and the scripts to rerun them.

## 1. Introduction

Embedding vectors are a good *computational* representation of meaning and a poor *data
format*: a vector from model A means nothing to model B, it cannot be read or grouped in SQL,
and every model upgrade forces re-embedding the corpus. A natural ambition is a shared,
stable *address* for meaning, like latitude and longitude: readable, hierarchical, frozen,
and the same whichever language or model produced the text.

This paper asks whether that ambition is achievable with current multilingual encoders and
simple, transparent machinery (linear projections, residual k-means, linear adapters). Our
answer is split:

* **Negative.** A *fine-grained* universal code (hundreds to thousands of top-level cells,
  stable across languages and models) is not supported by our measurements. The failure is
  structural, not a tuning issue: any partition of a continuous space splits equivalent
  texts near its boundaries, and the fraction split grows quickly with the number of cells.
* **Positive.** A *coarse* shared partition is stable enough to be useful, and the
  continuous canonical space behind it is an effective hub for cross-model migration.

We think the negative result is worth publishing on its own. "Semantic IDs" and shared
embedding spaces are active topics, and claims of a universal meaning code are easy to make
on in-sample numbers; ours looked good in-sample too (81% of training sentences get a
confident top-level code versus 26% held out).

The project name comes from Bhartṛhari's *sphoṭa* (5th c. CE): meaning that bursts forth whole
and stays the same across its utterances. Our results suggest it stays the same only at a
coarse resolution.

**Contributions.**
1. A complete, open method for a frozen hierarchical meaning code (§3).
2. A pre-registered decision gate, plus an amendment closing a granularity loophole (§4).
3. Full gate results across granularity, sample design and anchor (§5).
4. The granularity trade-off, the anchor effect and the language-collapse failure (§5–6).
5. Evidence for what does work: coarse regions and vector-based migration (§7).

## 2. Related work

**Semantic IDs.** TIGER (Rajput et al., 2023) quantizes item embeddings with an RQ-VAE into
codeword tuples for generative retrieval; DSI (Tay et al., 2022) uses hierarchical k-means
document IDs; Singh et al. (2024) study Semantic IDs for ranking. These codes are private to
one system and refit per catalogue, so they never face the cross-language and cross-model
stability test that a *universal* code must pass. Codebook collapse in RQ-VAE is mitigated by
EMA updates and dead-code revival (Zeghidour et al., 2021; Lee et al., 2022).

**Embedding translation.** vec2vec (Jha et al., 2025) translates between embedding spaces
without paired data, in support of the Platonic representation hypothesis (Huh et al.,
2024). Our results are compatible with that hypothesis at the level of *continuous*
geometry (migration works) while showing that *discrete* agreement is far weaker.

**Language-agnostic spaces.** LASER, LaBSE (Feng et al., 2022) and SONAR (Duquenne et al.,
2023) embed many languages jointly; Large Concept Models (2024) operate in SONAR space. We
quantify how much of their cross-lingual alignment survives discretization.

**Topic models.** BERTopic (Grootendorst, 2022) clusters embeddings and labels clusters with
c-TF-IDF, refitting per corpus. Our positive result is close to a *frozen, multilingual,
shared* BERTopic-style partition.

## 3. Method

### 3.1 Anchor encoder
`intfloat/multilingual-e5-small` (MIT, 384-d, ~100 languages), input `"query: " + text`, mean
pooling, L2 normalization. It was chosen in a bake-off against bge-m3, LaBSE and
paraphrase-multilingual-MiniLM for speed, size and an exact ONNX export (max abs diff 7·10⁻⁸
versus PyTorch), accepting slightly weaker translation alignment than LaBSE (0.67 vs 0.62 L1
agreement at 256 cells in the bake-off). §5.3 revisits this choice.

### 3.2 Canonical space (meaning-LDA)
With unit vectors `x`, `y = (x − μ) W`: PCA to 384 directions, then the generalized
eigenvectors of `(S_total, S_within + reg)`, where `S_within` is the scatter of translations of
the same sentence (FLORES-200 dev, 800 meanings × 12 languages; 7.2k Tatoeba pairs), shrunk
toward isotropy (`reg = 0.05`). The top 256 directions are scaled so within-meaning noise has
unit variance. As an ablation, PCA alone gives 34.1% cross-lingual L1 agreement versus 46.1%
with meaning-LDA (4096 cells).

### 3.3 Quantizer and code
Residual k-means with 4 levels (4096·256·256·256) on 77.6k sentences (Wikipedia sample in 12
languages + FLORES dev). Codes are Crockford base32 (widths 4/2/2/2) after a published affine
permutation, so 85% of single-symbol typos are rejected without a checksum.

### 3.4 Confidence
At each level, a softmax over the 16 nearest centroids' squared distances, with a temperature
fitted by maximum likelihood to predict the *held-out translation's* symbol; the product over
levels is the prefix confidence. Confidence drives truncation, alternates and the gate's
"confident portion".

### 3.5 Adapters
Other models enter the canonical space through ridge maps fitted on 40k paired texts
(registry: all-MiniLM-L6-v2, bge-small-en-v1.5, gte-small, paraphrase-multilingual-MiniLM-L12-v2,
LaBSE). An unsupervised vec2vec-style adapter failed at our CPU budget (top-1 retrieval 0%
versus 96.5% for ridge on the same texts), so supervised adapters are the default.

## 4. The decision gate

Owner review of the first release concluded that in-sample numbers did not support a
"GPS for meaning" claim. We therefore fixed the following gate **before** any v2 run (D-017).
All criteria use held-out FLORES-200 devtest sentences in 12 languages, never used to fit the
space, codebooks, calibration, labels or adapters.

| # | Criterion | Definition | Bar |
|---|---|---|---|
| 1 | Coverage | share of held-out sentences (all 12 languages) with L1 confidence ≥ 0.7 | ≥ 60% |
| 2 | Translation agreement | among en↔x pairs whose English code has L1 confidence ≥ 0.7, share with the same L1 code (mean over 11 languages) | ≥ 80% |
| 3 | Cross-model agreement | same English text, certified adapter vs anchor, same L1 code (mean over adapters) | ≥ 60% |
| 4 | Language collapse | per language, share of its sentences in L1 cells whose training members are ≥ 80% that language (English excluded) | ≤ 5% for every language |

**Amendment (D-018).** A tiny-scale dry run "passed" all four criteria with 16 top-level cells
and 6 languages. A gate that can be passed by making the partition coarse says nothing about
coordinates. We amended it: the gate must be met with **≥ 256 top-level cells**, on all 12
languages, with the cell count declared before the run. Results at 32–64 cells are reported
as evidence for a *coarse topic* product only.

## 5. Results

### 5.1 Full gate results

| Build | L1 cells | Coverage (≥ 60%) | Translation, confident (≥ 80%) | Cross-model (≥ 60%) | Worst collapse (≤ 5%) | Pass |
|---|---|---|---|---|---|---|
| v1 frozen release | 4096 | 29.4% | 65.1% | 15.7% | 33.8% (pt) | 0/4 |
| v1 re-clustered | 1024 | 42.6% | 69.7% | 23.1% | 44.2% (pt) | 0/4 |
| v1 re-clustered | 256 | 49.2% | 74.4% | 32.8% | 54.6% (pt) | 0/4 |
| v1 re-clustered | 128 | 54.1% | 73.6% | 44.6% | 0.1% | 1/4 |
| v1 re-clustered | 64 | 65.0% | 78.3% | 57.1% | 0.0% | 2/4 |
| v1 re-clustered | 32 | 69.0% | 79.3% | 64.5% | 0.0% | 3/4 |
| v2-mini (diversified sample) | 4096 | 30.9% | 65.0% | 16.3% | 26.8% (pt) | 0/4 |
| v2-mini | 32 | 63.9% | 78.8% | 62.3% | 0.1% | 3/4 |
| e5-small anchor (20k corpus) | 32 | 55.8% | 84.2% | 60.3%* | 0.0% | 3/4 |
| LaBSE anchor (20k corpus) | 32 | 66.5% | 88.7% | 53.0%* | 0.0% | 3/4 |
| LaBSE anchor (20k corpus) | 64 | 64.4% | 85.3% | 39.1%* | 0.0% | 3/4 |

\* Reduced cross-model panel (the other anchor and all-MiniLM-L6-v2).
Sources: `bench/results/l1-sweep.json`, `v2-mini-gate.json`, `v2-mini-4096-gate.json`,
`anchor-check.json`.

**No configuration passes all four criteria at any granularity, and none passes more than one
criterion at the amended bar of ≥ 256 cells.** The closest configurations (3 of 4) all use
32–64 top-level cells, which by the amendment cannot support a coordinate-system claim.

### 5.2 The granularity trade-off

Stability is bought with coarseness. Along the v1 re-clustering sweep, each 4× increase in
top-level cells costs roughly 10–25 points on the gate metrics:

| Step | Coverage | Translation (confident) | Cross-model |
|---|---|---|---|
| 32 → 128 | −14.9 | −5.7 | −20.0 |
| 64 → 256 | −15.8 | −3.9 | −24.3 |
| 256 → 1024 | −6.6 | −4.7 | −9.8 |
| 1024 → 4096 | −13.2 | −4.6 | −7.3 |

Cross-model agreement degrades fastest, then coverage; confident translation agreement
degrades more slowly because the confidence filter removes the hardest boundary cases (at the
price of coverage). The same pattern holds for LaBSE (32 → 64 cells: −2.1 coverage, −3.4
translation, −13.9 cross-model). We see no sign of a plateau: finer partitions simply place
more boundaries through regions where equivalent texts land.

### 5.3 The anchor effect: LaBSE vs e5-small

On the same 20k corpus at 32 cells, LaBSE beats e5-small on coverage (66.5% vs 55.8%, +10.7)
and confident translation agreement (88.7% vs 84.2%, +4.5), and is the only build that clears
the translation bar. It is worse on cross-model agreement with the reduced panel (53.0% vs
60.3%), plausibly because the panel's other models are closer to e5's geometry. The anchor is
therefore a first-order factor, comparable in size to one 4× granularity step, but it does not
change the conclusion: a better anchor buys roughly one step of resolution, not the six steps
from 32 to 4096 cells. bge-m3 (65% cross-lingual L1 agreement through its adapter) has not
been tested as an anchor; that is the one bounded GPU experiment we consider worth running,
under the amended bar.

### 5.4 Language collapse

The first release gave confident top-level codes to only 4.5% of Portuguese held-out sentences
versus 21–36% for other languages. Data, language tags, alignment (en→pt nearest neighbour
99.2%) and encoding were correct. The cause: **52.8% of Portuguese test sentences fell into
Portuguese-dominated cells** (80–96% Portuguese training members; 5 cells hold 53% of pt test
sentences), regardless of topic; for every other language the share was ≤ 0.2%
(`bench/results/language-collapse-small.json`). The Portuguese Wikipedia sample is unusually
homogeneous (mean pairwise cosine 0.84 vs ~0.78, dominated by sports and list stubs), so k-means
spent cells on *language identity* instead of meaning.

Two findings generalize:

* **Collapse is a function of granularity.** It is absent at ≤ 128 cells (≤ 0.1%) and appears
  abruptly at 256 cells (54.6% for pt). Once the partition is fine enough, residual language
  signal in the anchor becomes a cheaper way for k-means to reduce distortion than topic.
* **The sample explains only part of it.** A diversified per-language sample (≤ 40 sentences
  per title category, near-duplicates removed, ≤ 8000 per language) cut the worst collapse at
  4096 cells only from 33.8% to 26.8%. Part of the cause is the anchor itself: e5-small keeps a
  separable Portuguese direction that meaning-LDA does not fully remove.

Language collapse is invisible in average cross-lingual agreement and should be reported as a
separate metric by any multilingual quantization work.

## 6. Why fine-grained universal codes fail

Three effects compound. (1) **Boundary mass:** equivalent texts are not identical points but
small clouds (translation noise is our unit of distance); the fraction of clouds cut by some
cell boundary rises with the number of cells. (2) **Adapter error:** a linear adapter adds its
own displacement, which at fine granularity exceeds the cell radius; hence cross-model
agreement falls fastest. (3) **Residual language signal:** at fine granularity, k-means can
profitably spend cells on language rather than meaning. Confidence and alternates make (1)
visible but do not remove it; they trade coverage for precision. A code can be calibrated and
honest about its uncertainty and still not be *universal*.

## 7. What works

### 7.1 Coarse regions
At 32–64 top-level cells, confident translations share a region 79–89% of the time, coverage
is 64–69%, cross-model agreement is 53–65%, and there is no language collapse. That is a
useful frozen, multilingual, shared topic partition: readable region codes for analytics
dashboards, SQL `GROUP BY`, routing queries to specialised handlers, and semantic caching keys,
with finer levels kept as optional detail carrying explicit confidence.

### 7.2 Migration through stored canonical vectors
The continuous canonical space is a much better hub than the discrete code. Documents indexed
with model X through its adapter and queried with the anchor reach recall@10 of 96.5%
(all-MiniLM-L6-v2), 99.1% (bge-small-en-v1.5), 99.0% (gte-small) and 98.5%
(paraphrase-multilingual-MiniLM), versus 100% for re-embedding, and 98.3% for LaBSE in the
paraphrase setting; using codes alone gives only 10–28%. Storing the 256-d canonical vector
next to each item lets an organisation change embedding models without re-embedding its corpus,
with N adapters instead of N² pairwise maps.

### 7.3 Other measured properties of the frozen release
Cross-lingual L1 agreement 46.1% (stable mode 46.7%), paraphrase L1 agreement 71.6%; when both
codes have L1 confidence ≥ 0.7 (16% of pairs), translations agree 88.7%, at ≥ 0.9 (5%), 96.5%.
92% of L1 cells are used on the training corpus (perplexity 2182 of 4096). Codes alone are a
coarse index: translation retrieval top-1 30% (raw vectors 99.6%), SIB-200 topic P@10 16.6% vs
19.5%. Adding cross-model scatter to `S_within` helped at 1024 cells but not at 4096, while
lowering cross-lingual agreement; we kept it off.

## 8. Limitations

All experiments are CPU, `small` scale (≤ 80k fitting sentences, 12 languages, FLORES-200
devtest for evaluation). A full-scale GPU build with bge-m3 might move results by about one
granularity step; we expect, but have not shown, that it would still fail at ≥ 256 cells. The
cross-model panel for the anchor comparison is reduced. Labels for cells are machine-generated
and filtered, not natively reviewed. FLORES sentences are short, formal and topically narrow;
real traffic is likely harder.

## 9. Conclusion

Can meaning have universal coordinates? At fine resolution, with today's small multilingual
encoders and linear machinery, our pre-registered answer is **no**: no build passes a fixed,
held-out gate at ≥ 256 cells, every 4× refinement costs 10–25 points, and fine partitions
start encoding language instead of meaning. At coarse resolution the answer is a qualified
**yes**: a few dozen shared regions are stable across languages and reasonably stable across
models. And the continuous space behind the code is a practical hub for moving between models.
We recommend that work on semantic IDs and shared meaning spaces report held-out,
cross-language and cross-model agreement *as a function of granularity*, together with a
language-collapse metric, and fix the bar before running.

## Reproducibility

Gate definitions: `DECISIONS.md` D-017/D-018. Results: `bench/results/`. Experimental v2 build:
`experiments/v2-mini/`. GPU rerun: `bash scripts/gpu_build_v2.sh --scale full --sizes "256 1024 4096"`.
Code, artifacts and data recipes are Apache-2.0.

## References

Adelani et al. SIB-200, EACL 2024 · Chen et al. BGE M3, 2024 · Duquenne et al. SONAR, 2023 ·
Feng et al. LaBSE, ACL 2022 · Grootendorst. BERTopic, 2022 · Huh et al. The Platonic
Representation Hypothesis, ICML 2024 · Jha et al. Harnessing the Universal Geometry of
Embeddings, arXiv:2505.12540, 2025 · LCM team. Large Concept Models, 2024 · Lee et al. RQ-VAE,
CVPR 2022 · NLLB Team. FLORES-200, 2022 · Rajput et al. TIGER, NeurIPS 2023 · Singh et al.
Semantic IDs for ranking, 2024 · Tay et al. DSI, NeurIPS 2022 · Wang et al. Multilingual E5,
2024 · Zeghidour et al. SoundStream, 2021 · Bhartṛhari, Vākyapadīya.
