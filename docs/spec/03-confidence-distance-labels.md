# Sphota Specification v1, Part 3: Confidence, Precision, Distance and Labels

```
Status: Frozen (s1)
```

## 1. Confidence ("GPS accuracy")

For each level `l` the encoder considers the `M = 16` nearest centroids of the current
residual, with squared distances `d_1 ≤ d_2 ≤ … ≤ d_M`, and defines

```
p_l(k) = exp(−(d_k − d_1) / T_l) / Σ_j exp(−(d_j − d_1) / T_l)
level_confidence_l = p_l(chosen)
confidence_l       = Π_{j ≤ l} level_confidence_j          (cumulative)
```

The temperatures `T_l` are fitted by maximum likelihood on **held-out** pairs of
equivalent texts (translations from FLORES-200 dev meanings not used for the space, and
held-out Tatoeba pairs): `p_l` is trained to predict the *partner's* symbol given the same
prefix. Because the log-likelihood is a proper scoring rule, `confidence_l` is calibrated in
expectation: among codes with `confidence_2 ≈ 0.8`, about 80 % of translations share the L2
prefix. Reliability tables and expected calibration error (ECE) per level are published in
`calibration.json` and `docs/benchmark.md`.

**Semantics.** `confidence_l` estimates the probability that a different expression of the
same meaning (translation, paraphrase, or the same text through another certified encoder)
receives the same prefix of depth `l`.

## 2. Precision truncation

`encode(text, min_confidence=τ)` returns the longest prefix whose cumulative confidence is
`≥ τ`, and never less than one level. Truncated codes are ordinary prefixes. Like a GPS fix
with a wide accuracy radius, a short code is a correct-but-coarse answer.

## 3. Alternates and multi-code

* **Alternates.** For the first level whose top probability is below 0.98, the other
  candidates with `p ≥ 0.02` are returned as alternate codes (same depth as the returned
  code) with their probabilities. `is_boundary` is true when any returned level has
  `level_confidence < 0.6`.
* **Top-k codes.** A beam search over levels multiplies per-level probabilities and returns
  the `k` most probable full codes (for ambiguous text, or to index an item under several
  codes to improve recall). For alternative prefixes, deeper levels reuse the chosen path's
  symbols (approximation; exact rescoring is RECOMMENDED for k > 5).

## 4. Distance and similarity

A code (or prefix) `c` denotes the point `v(c) = Σ_{l ≤ depth(c)} C_l[i_l]`.

```
distance(a, b)   = ‖v(a) − v(b)‖₂                         (translation-noise units)
similarity(a, b) = clip( v(a)·v(b) / (‖v(a)‖ ‖v(b)‖), 0, 1 )
```

* Both need only the codebooks (numpy, no model).
* `neighbors(c, k)`: for an L1 code, the `k` nearest L1 centroids (precomputed table
  `neighbors.npz`, 32 per cell); for deeper codes, siblings under the same parent ranked by
  distance.
* `path(a, b)`: shortest path between the L1 cells of `a` and `b` on the 12-nearest-neighbour
  L1 graph with squared-distance edge weights (Dijkstra, ≤ 12 hops), ending at `b`. A
  "semantic route" for exploration and explanations.
* **Shared prefix depth** (`common_prefix`) is the purely symbolic distance; it is coarse
  but needs nothing at all.

## 5. Labels

`labels.json` maps keys to per-language strings:

* `"<i1>"` → label of the L1 cell,
* `"<i1>.<i2>"` → label of the L2 cell (present when the cell had ≥ 3 training members).

Labels are **c-TF-IDF keyphrases** (BERTopic-style) computed *per language* from the cell's
members in that language (Wikipedia + FLORES sentences). A Hindi label is made of Hindi
words from Hindi members: no machine translation is involved, so labels in different
languages are independent descriptions of the same region. When a language has no label for
a cell, implementations fall back to English, then to any available language.

A display label is `L1 label › L2 label`. Labels are **descriptive, not normative**: they
may be improved (typo fixes, better phrasing, more languages) without an RFC, but a label
MUST NOT be changed to describe a different region.
