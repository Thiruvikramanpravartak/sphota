# Gap analysis: what exists, what is missing, and how Sphota closes the gap

*Milestone R, 2026-11. Written before building v1 and kept up to date. Measured numbers live
in [`docs/benchmark.md`](../benchmark.md).*

## 1. The problem

Embedding vectors are the lingua franca of modern search, RAG, dedupe and analytics, but they
have three properties that make them bad as a *data format*:

1. **They are model-private.** A vector from model A means nothing to model B. Switching or
   upgrading the embedding model means re-embedding (re-indexing) every stored item.
2. **They are opaque.** `[0.013, -0.402, …]` cannot be read, grouped, counted, logged, put in a
   URL or a SQL `GROUP BY`.
3. **They are heavy.** 384–4096 floats per item, approximate-nearest-neighbour infrastructure to
   query them.

What is missing is a *coordinate system*: a stable, shared, human-readable way to say "this
text is about *here*", which survives model changes and language changes.

## 2. What exists

### 2.1 Semantic IDs in recommender systems
TIGER (Rajput et al., NeurIPS 2023, "Recommender Systems with Generative Retrieval") and
follow-ups quantize item embeddings with an RQ-VAE into short tuples of codewords ("Semantic
IDs"), so a generative model can decode item IDs token by token. Variants use residual k-means
(e.g. "Better Generalization with Semantic IDs", Singh et al., 2024) and hierarchical k-means
(DSI, Tay et al., 2022).

* **What they get right:** hierarchical discrete codes whose prefixes are coarser regions.
* **Gaps:** every system trains private codes on its own catalogue; the codes are not frozen,
  not shared, not human-readable, not multilingual by design, not portable across embedding
  models. The literature reports **codebook collapse** in plain RQ-VAE (a 2026 study reported
  only ~512 of 4096 coarse codes in use). Fixes include EMA codebook updates with dead-code
  revival (as in RQ-VAE variants such as RQP-VAE) or replacing the learned quantizer with
  (residual / hierarchical) k-means on a fixed embedding space.
* **Lesson for Sphota:** no single quantizer wins on every axis. Report *utilization*,
  *symbolic robustness* (do translations / paraphrases / other models get the same code?) and
  *geometric fidelity* (does code distance track meaning distance?) separately. We do.

### 2.2 Embedding translation / adapters
Commercial and open tools (e.g. EmbeddingAdapters, UniVec) learn pairwise dense maps between
two models' vector spaces so that a migration can avoid re-embedding.

* **Gaps:** pairwise (N models → N² maps), dense (still opaque floats), no shared reference
  frame, no standard or conformance test.
* **Lesson:** a *hub* design is better: every model maps into one canonical space (N maps).
  Cheap linear maps recover most of the signal when the models are similar.

### 2.3 Unsupervised translation between embedding spaces
vec2vec (Jha et al., 2025, "Harnessing the Universal Geometry of Embeddings",
arXiv:2505.12540) learns to translate between embedding spaces of different models *without
paired data*, supporting the "Platonic representation" hypothesis (Huh et al., ICML 2024):
strong models converge to similar geometry.

* **Relevance:** evidence that a universal reference space is feasible, and that adapters may
  eventually be trainable without paired data. vec2vec still outputs dense vectors in a
  target model's space; it does not define a shared, frozen coordinate system.

### 2.4 Language-agnostic concept spaces
Meta's SONAR (Duquenne et al., 2023) embeds 200 languages and speech into one space, and Large
Concept Models (LCM team, 2024) reason in that space. LaBSE (Feng et al., 2022) and LASER
(Artetxe & Schwenk, 2019) align sentences across languages.

* **Gaps:** continuous vectors tied to one model family; no discrete, human-readable codes;
  no frozen standard; no adapters for other models.

### 2.5 Topic models and taxonomies
BERTopic (Grootendorst, 2022) clusters embeddings and labels clusters with c-TF-IDF;
IAB / Google product taxonomies give human categories.

* **Gaps:** topic models are refit per corpus (labels and IDs change every run); taxonomies are
  hand-made, shallow, English-centric and not linked to embeddings.

## 3. What is missing (the gap)

| Property | Semantic IDs | Embedding adapters | vec2vec | SONAR / LCM | Topic models | **Sphota** |
|---|---|---|---|---|---|---|
| Open, frozen, versioned standard | – | – | – | – | – | **yes** |
| Discrete, human-readable codes | yes (opaque ints) | – | – | – | per-run IDs | **yes** (`7F3A-C2-91-0B`) |
| Hierarchical prefixes | yes | – | – | – | some | **yes, 4 levels** |
| Multilingual by design | – | – | – | yes | – | **yes (aligned before quantizing)** |
| Many models → one space | – | pairwise | pairwise, unpaired | one family | – | **hub adapters** |
| Per-level confidence / graceful truncation | – | – | – | – | – | **yes** |
| Labels in several languages | – | – | – | – | English keyphrases | **yes** |
| Conformance test + certificate | – | – | – | – | – | **yes** |
| Runs on CPU / in the browser | varies | varies | GPU | GPU | CPU | **yes** |

## 4. Design choices that close the gap

1. **Freeze a space, not a model.** v1 = anchor model → fixed projection → fixed codebook. The
   anchor is just the reference instrument; any model that maps into the space (via a
   certified adapter) produces conforming codes.
2. **Align languages before quantizing.** We fit a *meaning-LDA* projection: directions that
   maximize corpus variance relative to the variance among translations of the same sentence
   (FLORES-200 + Tatoeba). Distances are measured in "translation-noise units", so cell
   boundaries are placed where languages agree.
3. **Quantize with collapse-free residual k-means**, after comparing it with an RQ-VAE-style
   EMA quantizer (with dead-code revival) and a hierarchical k-means tree (bake-off below).
4. **Make uncertainty first-class (GPS accuracy).** Every level carries a calibrated
   confidence; `min_confidence` truncates unsure tails; near-boundary points return
   *alternates*. A shorter, right code beats a long, wrong one.
5. **Readable and checkable.** Crockford base32, fixed widths, optional version tag and check
   symbol; a sparse code space so most typos are detected.
6. **Govern it like a standard.** Spec written as an RFC, conformance thresholds, a certify
   command, a registry of adapters, and a frozen-forever v1 with migration tables for vN+1.

## 5. Honest limits (known before we start)

* Discretizing a continuous space always cuts some near-identical meanings apart. Agreement
  across languages/models is *probabilistic*, highest at L1 and lower at deeper levels.
  Sphota therefore reports agreement per level and exposes confidence rather than hiding it.
* The v1 space in this repository is built at `small` scale on CPU (tens of thousands of
  sentences, 12 languages). Its codebook is real and usable, but a `full`-scale rebuild on a
  GPU (millions of sentences, 100+ languages) is the intended v1.0 final.
* English-only models can be adapted into the space, but they cannot place non-English text
  well; certificates say so.
* Codes reveal topics. They are not anonymization; see the privacy section of the spec.

## 6. Bake-off (milestone R)

See [`DECISIONS.md`](../../DECISIONS.md) D-003…D-006 and
[`bench/results/bakeoff-tiny.json`](../../bench/results/bakeoff-tiny.json).

## References

* Rajput et al. *Recommender Systems with Generative Retrieval.* NeurIPS 2023.
* Tay et al. *Transformer Memory as a Differentiable Search Index.* NeurIPS 2022.
* Singh et al. *Better Generalization with Semantic IDs: A Case Study in Ranking for
  Recommendations.* 2024.
* Lee et al. *Autoregressive Image Generation using Residual Quantization (RQ-VAE).* CVPR 2022.
* Zeghidour et al. *SoundStream* (residual VQ with EMA and codebook re-initialization). 2021.
* Jha, Zhang, Shmatikov, Morris. *Harnessing the Universal Geometry of Embeddings.*
  arXiv:2505.12540, 2025.
* Huh, Cheung, Wang, Isola. *The Platonic Representation Hypothesis.* ICML 2024.
* Duquenne, Schwenk, Sagot. *SONAR: Sentence-Level Multimodal and Language-Agnostic
  Representations.* 2023.
* LCM team. *Large Concept Models: Language Modeling in a Sentence Representation Space.* 2024.
* Feng et al. *Language-agnostic BERT Sentence Embedding (LaBSE).* ACL 2022.
* Chen et al. *BGE M3-Embedding.* 2024.
* Wang et al. *Multilingual E5 Text Embeddings: A Technical Report.* 2024.
* NLLB Team. *No Language Left Behind (FLORES-200).* 2022.
* Adelani et al. *SIB-200.* EACL 2024.
* Grootendorst. *BERTopic.* 2022.
* Crockford. *Base32 encoding.* https://www.crockford.com/base32.html
