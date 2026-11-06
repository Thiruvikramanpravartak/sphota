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

## Anchor check: multilingual-e5-small vs LaBSE (D-017 metrics)

Same corpus for both anchors (20k shared Wikipedia/FLORES texts + FLORES dev[:800] × 12 languages),
same meaning-LDA, same L1 sizes; cross-model panel = the other anchor + all-MiniLM-L6-v2 (CPU budget).
Absolute numbers differ from the L1 sweep above because the clustering corpus is different.

| anchor | L1 size | coverage | translation, confident | top-2 translation | cross-model | collapse | gate (1/2/3/4) |
|---|---|---|---|---|---|---|---|
| multilingual-e5-small | 32 | 55.8% | 84.2% | 87.1% | 60.3% | 0.0% | ✗/✓/✓/✓ |
| multilingual-e5-small | 64 | 49.2% | 81.1% | 82.6% | 49.2% | 0.0% | ✗/✓/✗/✓ |
| LaBSE | 32 | 66.5% | 88.7% | 91.3% | 53.0% | 0.0% | ✓/✓/✗/✓ |
| LaBSE | 64 | 64.4% | 85.3% | 89.0% | 39.1% | 0.0% | ✓/✓/✗/✓ |

**Reading:** LaBSE gives clearly better coverage and cross-lingual agreement; models map into its
space less well. A larger anchor (bge-m3) on a GPU is the open question.

### Deciding CPU experiment: LaBSE anchor at ≥ 256 cells (D-018-compliant)

| anchor | L1 cells | coverage | translation, confident | cross-model* | collapse | gate (1/2/3/4) |
|---|---|---|---|---|---|---|
| LaBSE | 256 | 64.0% | 77.2% | 26.1% | 0.0% | ✓/✗/✗/✓ |
| LaBSE | 1024 | 79.6% | 79.1% | 50.7% | 0.0% | ✓/✗/✗/✓ |

\* panel: e5-small + all-MiniLM-L6-v2 mapped into the LaBSE space (20k corpus). No ≥ 256-cell configuration passes; translation agreement
is within ~1 point of the bar at 1024 cells, cross-model agreement is the binding constraint.

## v2-mini (diversified sample, CPU)

| L1 size | coverage | translation, confident | cross-model | collapse | gate |
|---|---|---|---|---|---|
| 32 | 63.9% | 78.8% | 62.3% | 0.1% | ✓/✗/✓/✓ |
| 64 | 55.3% | 78.9% | 53.5% | 0.1% | ✗/✗/✗/✓ |
| 4096 | 30.9% | 65.0% | 16.3% | 26.8% (pt) | ✗/✗/✗/✗ |

At 4096 cells the diversified sample reduces the worst collapse (33.8% → 26.8%) but does not fix it.
See DECISIONS D-018: the gate is only meaningful with ≥ 256 top-level cells.

## Indian languages (FLORES-200 devtest, 1012 sentences, `python -m sphota.build.indic`)
Each language against its English original, frozen space s1, confidence threshold 0.7
(`bench/results/indic-small.json`).

| language | region agrees with English | region coverage ≥ 0.7 | agreement when both confident | L1 agrees | retrieval@1 | top-region share |
|---|---|---|---|---|---|---|
| English (reference) | | 68.4% | | | | 36.1% (R17) |
| Hindi (`hi`) | 76.4% | 68.6% | 92.2% | 52.7% | 99.9% | 39.8% (R17) |
| Bengali (`bn`) | 71.9% | 69.0% | 89.1% | 47.3% | 100.0% | 41.0% (R17) |
| Marathi (`mr`) | 72.4% | 69.3% | 89.5% | 48.2% | 99.5% | 39.7% (R17) |
| Telugu (`te`) | 71.3% | 66.5% | 90.1% | 44.7% | 99.6% | 37.5% (R17) |
| Tamil (`ta`) | 73.1% | 69.4% | 90.4% | 45.0% | 99.3% | 39.3% (R17) |
| Urdu (`ur`) | 72.0% | 69.3% | 88.3% | 45.1% | 99.5% | 42.1% (R17) |
| Gujarati (`gu`) | 71.8% | 66.9% | 87.8% | 46.5% | 99.6% | 39.7% (R17) |
| Kannada (`kn`) | 73.7% | 68.7% | 90.7% | 50.3% | 99.2% | 39.9% (R17) |
| Odia (`or`) | 72.0% | 70.3% | 86.8% | 45.1% | 99.4% | 40.8% (R17) |
| Malayalam (`ml`) | 72.4% | 69.6% | 91.1% | 47.8% | 99.7% | 37.6% (R17) |
| Punjabi (`pa`) | 70.2% | 67.1% | 87.5% | 43.3% | 99.1% | 38.6% (R17) |
| Assamese (`as`) | 67.4% | 70.9% | 82.7% | 37.0% | 97.9% | 45.1% (R17) |
| Maithili (`mai`) | 71.0% | 70.0% | 89.2% | 46.2% | 99.8% | 43.5% (R17) |
| Nepali (`ne`) | 71.2% | 68.8% | 89.7% | 47.0% | 99.2% | 40.8% (R17) |
| Sindhi (`sd`) | 62.5% | 67.5% | 78.7% | 34.8% | 98.3% | 47.8% (R17) |
| Sanskrit (`sa`) | 63.2% | 67.0% | 82.2% | 34.1% | 95.7% | 52.3% (R17) |
| Kashmiri (Arabic) (`ks`) | 61.1% | 69.7% | 77.9% | 30.4% | 91.7% | 49.3% (R17) |
| Manipuri (Bengali script) (`mni`) | 45.4% | 75.7% | 55.6% | 15.8% | 62.0% | 68.6% (R17) |
| Santali (Ol Chiki) (`sat`) | 36.1% | 94.2% | 34.9% | 1.0% | 2.6% | 97.5% (R17) |

* **Works well (13 languages):** Hindi, Bengali, Marathi, Telugu, Tamil, Urdu, Gujarati, Kannada, Odia,
  Malayalam, Punjabi, Maithili, Nepali. When both sides are confident, the region matches English
  87–92% of the time; the canonical vector finds the English original 99%+ of the time out of 1012.
* **Partial:** Assamese, Sindhi, Sanskrit, Kashmiri (78–83% confident agreement, 92–98% retrieval).
* **Not supported:** Manipuri (Bengali script) and Santali (Ol Chiki). Santali is *confidently wrong*:
  94% of its sentences get region confidence ≥ 0.7, but 97.5% land in one region (R17), and retrieval
  is 2.6%. Do not use Sphota codes for these languages.
* Region names are translated (machine, unreviewed) for te, kn, ml, gu, or, as, ur; Punjabi and
  Nepali names fall back to English (D-021).

## GPU gate, medium scale (interim, Kaggle T4, D-022)
| anchor | L1 regions | coverage ≥ 60% | translation (confident) ≥ 80% | cross-model ≥ 60% | collapse ≤ 5% |
|---|---|---|---|---|---|
| e5-small | (kept) | | | 16.2% FAIL | 34.2% FAIL |
| LaBSE | 256 | 69.4% PASS | 83.5% PASS | 40.1% FAIL | 0.2% PASS |
| LaBSE | 1024 | 55.2% FAIL | 83.9% PASS | 29.4% FAIL | 0.1% PASS |

LaBSE at 4096 and the bge-m3 anchor were still running; `bench/results/kaggle-medium-interim.json`.

## Limitations

* `small`-scale build: ~78k sentences in 12 languages fit 4096 L1 cells (~19 per cell). Regions for
  rare topics are coarse; a GPU `full`-scale rebuild (s2) is planned.
* The anchor (multilingual-e5-small) is small; very short or idiomatic sentences in some languages
  (e.g. 4-word Hindi or French sentences) get low confidence and sometimes different L1 regions.
* Cross-model agreement at the *exact code* level is modest. For model migration rely on prefix
  search plus alternates, and expect recall rather than identity.
* Paraphrase data is English only (STS-B). Cross-lingual numbers use FLORES (news/wiki style).
* Labels are automatic (LLM names + machine translation) and can be wrong.
