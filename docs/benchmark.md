# Benchmarks

All numbers are measured on data **never used** to fit the space, codebooks,
calibration or adapters: FLORES-200 *devtest* (1012 sentences × 12 languages), SIB-200
topic labels for those sentences, and STS-B paraphrase pairs scored ≥ 4.5/5.
Space `s1`, built at `small` scale on CPU. Raw results:
[`bench/results/`](https://github.com/thiruvikramanpravartak/sphota/tree/main/bench/results).
Reproduce: `python -m sphota.build.bench --scale small`.

!!! note "How to read agreement"
    *Agreement at L2* = fraction of pairs whose codes share the first two levels. A random
    pair shares L1 with probability ≈ 1/4096. Discretizing a continuous space always
    splits some equivalent texts across a boundary, so agreement falls with depth;
    confidence (below) tells you which codes to trust.

## 1. Cross-lingual agreement (anchor)

English sentence vs. its human translation, same code prefix:

| | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| **mean over 11 languages** | 46.1% | 22.9% | 10.9% | 4.8% |
| all language pairs | 42.1% | 19.1% | 8.2% | 3.4% |
| stable mode | 46.7% | 23.4% | 11.2% | 5.1% |
| en ↔ hi | 52.7% | 27.6% | 13.6% | 6.4% |
| en ↔ ta | 45.0% | 21.2% | 8.9% | 3.5% |
| en ↔ es | 55.6% | 31.1% | 15.9% | 7.9% |
| en ↔ fr | 50.0% | 27.1% | 12.9% | 6.0% |
| en ↔ de | 56.2% | 31.1% | 17.6% | 8.0% |
| en ↔ zh | 45.5% | 21.7% | 8.2% | 3.2% |
| en ↔ ar | 38.4% | 17.1% | 7.5% | 2.7% |
| en ↔ ru | 51.4% | 27.6% | 14.2% | 7.2% |
| en ↔ ja | 46.3% | 18.9% | 7.9% | 2.4% |
| en ↔ pt | 19.1% | 7.4% | 3.1% | 1.0% |
| en ↔ bn | 47.3% | 21.5% | 10.6% | 4.8% |

Effect of cross-lingual alignment (design sweep, same sizes): L1 34.1% with plain
PCA-whitening → 46.2% with meaning-LDA; L2 8.3% → 22.2%.

## 2. Paraphrase stability (English, STS-B ≥ 4.5)

918 pairs.

| | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| default | 71.6% | 53.8% | 38.3% | 27.8% |
| stable mode | 71.7% | 54.4% | 38.2% | 28.2% |

## 3. Precision truncation (GPS accuracy)

With `min_confidence=τ`, codes are cut where the cumulative confidence drops below τ.
*Consistent* = the shorter code of the pair is a prefix of the longer one (they do not
contradict each other).

| τ | consistent (cross-lingual) | mean depth | consistent (paraphrase) | mean depth |
|---|---|---|---|---|
| 0.0 | 4.8% | 4.00 | 27.8% | 4.00 |
| 0.3 | 39.2% | 1.74 | 62.6% | 2.06 |
| 0.5 | 44.7% | 1.28 | 69.9% | 1.51 |
| 0.6 | 45.6% | 1.17 | 71.0% | 1.33 |
| 0.7 | 45.9% | 1.09 | 71.4% | 1.23 |
| 0.8 | 46.1% | 1.05 | 71.5% | 1.15 |
| 0.9 | 46.1% | 1.02 | 71.6% | 1.07 |

### Confidence coverage (how many sentences get a confident code)

Share of held-out FLORES devtest sentences whose L1 confidence is ≥ τ: τ=0.5 → 46.7%,
τ=0.7 → **26.0%** (English 36.0%, Hindi 35.4%, Tamil 29.9%, Chinese 21.0%, Arabic 21.1%,
Portuguese 4.5%: 53% of Portuguese sentences fall into Portuguese-only cells formed from a homogeneous training sample, see DECISIONS D-016; 27.9% excluding Portuguese), τ=0.9 → 9.5%. L2 ≥ 0.7: 5.9%. STS-B English
sentences: 38.9% at τ=0.7. On the *training* corpus the same share is 81%, so confidence is
much higher in-sample than on new text. One-sided: when the English code has L1 confidence
≥ 0.7, its translation shares the L1 code 66.7% of the time (≥ 0.9: 73.5%, coverage 14.5%).
**Reading:** confident codes are reliable but rare at `small` scale; most new sentences get a
low-confidence code.

### Calibration on unseen data

| level | ECE | bins (confidence → observed agreement) |
|---|---|---|
| L1 | 0.093 | 0.17→0.12 (n=1175); 0.30→0.28 (n=5941); 0.49→0.41 (n=6063); 0.70→0.56 (n=4360); 0.91→0.75 (n=4725) |
| L2 | 0.099 | 0.12→0.10 (n=7640); 0.29→0.21 (n=7731); 0.48→0.32 (n=4036); 0.68→0.43 (n=1807); 0.88→0.57 (n=1050) |
| L3 | 0.085 | 0.10→0.07 (n=14069); 0.28→0.15 (n=5512); 0.48→0.22 (n=2015); 0.68→0.36 (n=534); 0.86→0.38 (n=134) |
| L4 | 0.060 | 0.07→0.04 (n=18925); 0.28→0.10 (n=2471); 0.47→0.17 (n=708); 0.69→0.35 (n=138); 0.86→0.18 (n=22) |

## 4. Utilization (no codebook collapse)

| set | n | L1 codes used | L1 perplexity (of 4096) | largest cell share | L2 / L3 / L4 symbols used |
|---|---|---|---|---|---|
| training corpus | 68,000 | 3754 (91.6%) | 2181.9 | 0.3% | 100.0% / 100.0% / 100.0% |
| held out | 13,980 | 806 (19.7%) | 342.8 | 2.8% | 19.5% / 19.9% / 14.1% |

## 5. Retrieval with codes vs. raw vectors

### Translation retrieval (English query → find its translation among 1012, mean of 11 languages)

| representation | top-1 | recall@10 |
|---|---|---|
| raw anchor vectors (384-d floats) | 99.6% | 100.0% |
| canonical vectors (256-d floats) | 99.4% | 100.0% |
| **codes only** (13 characters) | 30.3% | 55.5% |
| codes + L1 alternates (multi-probe) | 35.6% | 68.3% |

### Topic retrieval (SIB-200, 7 topics, n=92): precision@10 of same-topic neighbours

| representation | P@10 |
|---|---|
| raw anchor vectors | 19.5% |
| canonical vectors | 21.1% |
| codes only | 16.6% |
| codes, English queries → Hindi documents | 16.9% |
| raw vectors, English → Hindi | 20.9% |

Codes are a coarse index: they win on size, readability, and portability, and lose fine
ranking precision. Use them to filter/organize, and re-rank with vectors when you have them.

## 6. Cross-model agreement (adapters)

Same text, model + adapter vs. the anchor (FLORES devtest). *Reference* models were used
when fitting the space (their disagreement is treated as noise); held-out models were not.

| model | role | L1 (en) | L2 (en) | L1 (all langs) | cross-lingual L1 (model) | fidelity (cos) | grade |
|---|---|---|---|---|---|---|---|
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | not used to fit the space | 23.3% | 5.8% | 24.5% | 54.8% | 0.580 | certified-multilingual |
| `BAAI/bge-m3` | not used to fit the space | 19.3% | 5.3% | 22.7% | 65.3% | 0.629 | not-certified |
| `sentence-transformers/LaBSE` | not used to fit the space | 19.2% | 6.0% | 19.4% | 56.3% | 0.602 | certified-multilingual |
| `sentence-transformers/all-MiniLM-L6-v2` | not used to fit the space | 16.6% | 3.1% | 5.2% | 18.1% | 0.548 | certified |
| `BAAI/bge-small-en-v1.5` | not used to fit the space | 13.8% | 2.4% | 4.9% | 22.1% | 0.568 | certified |
| `thenlper/gte-small` | not used to fit the space | 5.7% | 0.5% | 3.4% | 30.4% | 0.548 | certified |

Anchor int8 ONNX variant vs fp32 reference (same text): 84.8% | 74.6% | 66.8% | 59.8% (L1..L4).

## 8. Unsupervised adapter (vec2vec-style, no paired data)

`sentence-transformers/all-MiniLM-L6-v2` → s1 with disjoint text sets (20,000 per side), 2500 steps on CPU.

| | top-1 retrieval (of 1012) | mean cosine | L1 agreement | L2 agreement |
|---|---|---|---|---|
| unsupervised | 0.0% | 0.004 | 0.4% | 0.0% |
| supervised ridge (same budget) | 96.5% | 0.544 | 22.3% | 5.7% |
| chance | 0.1% | 0 | ≈0 | ≈0 |

## 9. Speed (CPU)

* Encode, ONNX fp32: **155.3 sentences/s**; int8: 331.0 sentences/s (4 vCPU (CI container)).
* Quantize precomputed vectors (numpy): 10,331 vectors/s.


## L1 granularity sweep (D-017 gate metrics)

Top level re-clustered on the frozen v1 canonical vectors (no re-embedding); confidence temperature
re-calibrated per size on the same held-out calibration pairs. FLORES-200 devtest, 12 languages.
Cross-model = mean over the 5 registry adapters (same English text, same L1 cell as the anchor).

| L1 size | coverage (conf ≥ 0.7) | translation, all | translation, confident | top-2 translation | cross-model | worst language collapse | gate (1/2/3/4) |
|---|---|---|---|---|---|---|---|
| 32 | 69.0% | 70.5% | 79.3% | 88.5% | 64.5% | 0.0% (hi) | ✓/✗/✓/✓ |
| 64 | 65.0% | 70.7% | 78.3% | 86.3% | 57.1% | 0.0% (hi) | ✓/✗/✗/✓ |
| 128 | 54.1% | 62.9% | 73.6% | 79.2% | 44.6% | 0.1% (ar) | ✗/✗/✗/✓ |
| 256 | 49.2% | 61.4% | 74.4% | 76.6% | 32.8% | 54.6% (pt) | ✗/✗/✗/✗ |
| 1024 | 42.5% | 55.3% | 69.7% | 71.2% | 23.1% | 44.2% (pt) | ✗/✗/✗/✗ |
| 4096 | 29.4% | 46.1% | 65.1% | 61.4% | 15.7% | 33.8% (pt) | ✗/✗/✗/✗ |

**Reading:** stability is bought with coarseness. Only very coarse top levels (32–64 regions)
approach the gate; at 4096 regions (v1) all four criteria fail. The language-collapse criterion uses
the full training corpus (Wikipedia + FLORES dev) to decide which cells are language-dominated, so the
4096 row (33.8%, Portuguese) differs slightly from the Wikipedia-only figure in D-016 (52.8%).

## Limitations

* `small`-scale build: ~78k sentences in 12 languages fit 4096 L1 cells (~19 per cell). Regions for
  rare topics are coarse; a GPU `full`-scale rebuild (s2) is planned.
* The anchor (multilingual-e5-small) is small; very short or idiomatic sentences in some languages
  (e.g. 4-word Hindi or French sentences) get low confidence and sometimes different L1 regions.
* Cross-model agreement at the *exact code* level is modest. For model migration rely on prefix
  search plus alternates, and expect recall rather than identity.
* Paraphrase data is English only (STS-B). Cross-lingual numbers use FLORES (news/wiki style).
* Labels are automatic (LLM names + machine translation) and can be wrong.
