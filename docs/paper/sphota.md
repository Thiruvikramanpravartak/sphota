# Sphota: An Open, Frozen, Human-Readable Coordinate System for Meaning

*Draft for arXiv (cs.CL; cross-list cs.IR). The Sphota Authors, 2026.*

## Abstract

Embedding vectors are the de facto representation of meaning in search, retrieval-augmented
generation, deduplication and analytics, yet they are model-private, opaque and must be
recomputed whenever the model changes. We present **Sphota**, an open and *frozen* coordinate
system for meaning: a fixed 256-dimensional canonical space derived from a small multilingual
anchor encoder, and a four-level residual codebook (4096·256·256·256) that maps any text to a
short human-readable code such as `JYZA-7M-X0-A8`. Languages are aligned *before*
quantization with a meaning-LDA projection fitted on parallel data, raising cross-lingual
first-level agreement from 34% to 46% (12 languages, 4096 cells). Every code level carries a
confidence calibrated by maximum likelihood on held-out translations, enabling GPS-style
precision truncation, alternates for near-boundary inputs and multi-code assignment. Other
embedding models enter the space through linear adapters; storing the 256-d canonical vector
next to the code lets an index built with one model be queried with another at 96.5–99.1%
recall@10 (paraphrase retrieval), without re-embedding the corpus. We release the
specification, a conformance procedure with certificates, the frozen artifacts, a numpy-only
core library, integrations, and an in-browser "Meaning Atlas". We report negative results
and limitations openly: exact code identity across models is modest (6–23% at L1), and a
cross-model whitening term that helped at 1024 cells did not help at 4096.

## 1. Introduction

Vectors are an excellent *computational* representation of meaning but a poor *data format*.
A vector from model A is meaningless to model B, so a model upgrade forces re-embedding every
stored item; a vector cannot be read, grouped in SQL or placed in a URL; and every organization
trains or adopts its own private space. We ask: can meaning have a shared, stable *address*,
like latitude and longitude, readable by people and reachable from many models?

The name comes from Bhartṛhari's *sphoṭa* (5th c. CE): the meaning that bursts forth as a whole
and remains the same across its many utterances.

**Contributions.**
1. A frozen canonical space with cross-lingual alignment before quantization (§3).
2. A hierarchical, human-readable code format with typo detection and versioning (§3.4).
3. Calibrated per-level confidence, precision truncation, alternates and stable mode (§3.5).
4. A hub of adapters with recall-oriented conformance certification (§3.6).
5. An open benchmark measuring cross-lingual, paraphrase and cross-model agreement,
   utilization, retrieval and migration separately (§4), including negative results.

## 2. Related work

**Semantic IDs.** TIGER (Rajput et al., 2023) quantizes item embeddings with an RQ-VAE into
codeword tuples decoded by a generative retriever; DSI (Tay et al., 2022) uses hierarchical
k-means document IDs; Singh et al. (2024) study Semantic IDs for ranking. These codes are
private to one system and refit per catalogue; plain RQ-VAE suffers codebook collapse, mitigated
by EMA updates and dead-code revival (Zeghidour et al., 2021; Lee et al., 2022) or by k-means.
Sphota borrows the residual hierarchy but freezes and publishes it, aligns languages first, and
evaluates utilization separately from robustness.

**Embedding translation.** Commercial adapters learn pairwise maps between embedding spaces.
vec2vec (Jha et al., 2025) translates between spaces without paired data, supporting the
Platonic representation hypothesis (Huh et al., 2024). Sphota uses a hub (N maps instead of N²)
and evaluates an unsupervised adapter at small scale (§4.7).

**Language-agnostic spaces.** LASER, LaBSE (Feng et al., 2022) and SONAR (Duquenne et al.,
2023) embed many languages jointly; Large Concept Models (2024) operate in SONAR space. These
are continuous and model-family-specific; Sphota contributes discrete, readable, governed
coordinates.

**Topic models.** BERTopic (Grootendorst, 2022) clusters embeddings and labels clusters with
c-TF-IDF but refits per corpus; Sphota labels are computed once for a frozen partition.

## 3. Method

### 3.1 Anchor
`intfloat/multilingual-e5-small` (MIT, 384-d, ~100 languages), input `"query: " + text`, mean
pooling, L2 normalization. It was chosen in a bake-off against bge-m3, LaBSE and
paraphrase-multilingual-MiniLM: LaBSE aligned translations slightly better (0.67 vs 0.62 L1
agreement at 256 cells) but is 4× slower and 1.9 GB; e5-small has an official fp32 ONNX export
(bit-exact with PyTorch, max abs diff 7·10⁻⁸) and runs in browsers.

### 3.2 Canonical space (meaning-LDA)
With unit vectors `x`, we compute PCA to `P = 384` directions, then the generalized
eigenvectors of `(S_total, S_within + reg)` where `S_within` is the scatter of translations of
the same sentence (FLORES-200 dev, 800 meanings × 12 languages; 7.2k Tatoeba pairs), shrunk
toward isotropy (`reg = 0.05`). The top 256 directions are scaled so that within-meaning noise
has unit variance: distance is measured in translation-noise units.

### 3.3 Quantizer
Residual k-means with 4 levels (4096, 256, 256, 256), fitted on 77.6k sentences (Wikipedia
sample in 12 languages + FLORES dev). In a tiny-scale bake-off, an RQ quantizer with EMA and
dead-code revival used fewer effective codes (perplexity 0.19–0.23·K vs 0.29–0.37·K), and a
hierarchical k-means tree degenerated at deep levels with little data.

### 3.4 Code format
Crockford base32 with widths 4/2/2/2 (20/10/10/10 bits). Indices pass through a published
affine permutation `s = (a·i + b) mod 2^bits`, so only 4096 of 2²⁰ L1 symbols are valid and 85%
of single-symbol typos are rejected without a checksum; an optional mod-37 check symbol and an
`s1:` version tag are defined.

### 3.5 Confidence, truncation, alternates
At each level, a softmax over the 16 nearest centroids' squared distances with a temperature
fitted by maximum likelihood to predict the *held-out translation's* symbol given the same
prefix (a proper scoring rule) yields calibrated conditional confidences; their product is the
prefix confidence. `min_confidence` truncates; alternates are re-quantized exactly under the
alternative prefix; a beam search returns top-k codes. Stable mode is a deterministic majority
vote of nearest centroids under 32 fixed Gaussian perturbations.

### 3.6 Adapters and conformance
A ridge map from a model's normalized vectors to canonical vectors is fitted on 40k paired
texts; a per-adapter temperature scale is fitted on held-out pairs. Certification measures
neighbourhood agreement (model L1 ∈ anchor L1 ∪ its 8 nearest cells), top-3 agreement,
fidelity, and exact agreement (reported, not gated).

## 4. Experiments

All test data is held out: FLORES-200 devtest (1012 × 12 languages), SIB-200 topics, STS-B
pairs with score ≥ 4.5 (918).

| | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| Cross-lingual agreement (en ↔ 11 languages) | 46.1% | 22.9% | 11.0% | 4.8% |
| …stable mode | 46.7% | 23.4% | 11.2% | 5.1% |
| Paraphrase agreement (English) | 71.6% | 53.8% | 38.3% | 27.8% |
| PCA-only space, cross-lingual (ablation) | 34.1% | 8.3% | 2.6% | 0.9% |

**Utilization.** 92% of L1 cells used on the training corpus, perplexity 2182 of 4096, the
largest cell holds 0.29% of data.

**Retrieval.** Codes alone: translation retrieval top-1 30% (raw vectors 99.6%); topic
retrieval P@10 16.6% vs 19.5% for raw vectors (SIB-200, 7 classes). Codes are a coarse index.

**Migration.** Documents indexed with model X via its adapter, queried with the anchor:
recall@10 96.5% (all-MiniLM-L6-v2), 99.1% (bge-small-en-v1.5), 99.0% (gte-small), 98.5%
(paraphrase-multilingual-MiniLM) using canonical vectors, versus 100% for re-embedding;
codes only: 10–28%.

**Cross-model agreement.** Exact L1 agreement with the anchor on the same English text:
6–23%; neighbourhood agreement: 62–64% (chance 0.2%).

**Negative result.** Adding cross-model residual scatter to `S_within` improved adapter
agreement at 1024 L1 cells (held-out model 0.18 → 0.21 at λ=1) but not at 4096 (0.166 vs 0.167)
while lowering cross-lingual agreement (0.462 → 0.450). We kept λ = 0.

**Unsupervised adapter.** See `bench/results/vec2vec-small.json` and docs/benchmark.md §8.

## 5. Discussion and limitations

Any partition of a continuous space splits some equivalent texts; Sphota makes the uncertainty
explicit rather than hiding it. The small anchor limits cross-lingual agreement for very short
or idiomatic sentences. The released build is `small` scale (CPU); the method is designed for a
`full`-scale GPU rebuild as space `s2` with a migration table. Codes reveal topics and are not
anonymization; we document privacy by depth. Labels are generated by a small LLM and machine
translation and may be wrong.

## 6. Conclusion

A frozen, open coordinate system for meaning is practical today on a CPU: readable codes for
organizing, filtering, routing and sharing; a model-independent canonical vector for precise
cross-model search; and governance that keeps both stable.

## References

Adelani et al. SIB-200, EACL 2024 · Duquenne et al. SONAR, 2023 · Feng et al. LaBSE, ACL 2022 ·
Grootendorst. BERTopic, 2022 · Huh et al. The Platonic Representation Hypothesis, ICML 2024 ·
Jha et al. Harnessing the Universal Geometry of Embeddings, arXiv:2505.12540, 2025 · LCM team.
Large Concept Models, 2024 · Lee et al. RQ-VAE, CVPR 2022 · NLLB Team. FLORES-200, 2022 ·
Rajput et al. TIGER, NeurIPS 2023 · Singh et al. Semantic IDs for ranking, 2024 · Tay et al.
DSI, NeurIPS 2022 · Wang et al. Multilingual E5, 2024 · Zeghidour et al. SoundStream, 2021 ·
Chen et al. BGE M3, 2024 · Bhartṛhari, Vākyapadīya.
