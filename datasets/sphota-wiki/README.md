---
license: cc-by-sa-4.0
language: [en, hi, ta, es, fr, de, zh, ar, ru, ja, pt, bn]
pretty_name: Sphota-tagged Wikipedia sample (space s1)
size_categories: [10K<n<100K]
task_categories: [text-classification, sentence-similarity]
tags: [sphota, semantic-ids, multilingual, topic-modeling, wikipedia]
---

# Sphota-tagged Wikipedia sample (s1, small)

68,000 Wikipedia sentences in 12 languages, each tagged with its
[Sphota](https://github.com/thiruvikramanpravartak/sphota) s1 code (a frozen, hierarchical
"coordinate of meaning"), per-level confidence and the English label of its L1 region.

| column | description |
|---|---|
| `text` | sentence (Wikipedia 20231101 dump) |
| `lang` | ISO 639-1 code |
| `wiki_title` | source article title |
| `sphota` | full code, e.g. `JYZA-7M-X0-A8` |
| `sphota_l1`, `sphota_l2` | prefixes (topic, sub-topic) |
| `conf_l1` … `conf_l4` | calibrated cumulative confidence per level |
| `label_en` | English label of the L1 region |
| `space` | `s1` |

**Composition:** en 24,000; hi, ta, es, fr, de, zh, ar, ru, ja, pt, bn 4,000 each. Sentences
sampled from random row groups of random shards, ≤ 3 per article, 25–300 characters,
mostly-alphabetic filter. 3,754 distinct L1 codes, 38,270 distinct L2 prefixes.

**Caveat:** this sample is (part of) the corpus the s1 codebook was fitted on, so codes are
in-sample. For held-out evaluation use FLORES-200 devtest (see the Sphota benchmark).

**Uses:** multilingual topic analytics, cross-lingual grouping by prefix, weak topic labels,
studying the Sphota space. **Not for:** inferring facts about people; codes reveal topics.

**Reproduce / scale up:** `python -m sphota.build.data --scale small && python -m
sphota.build.open_data --scale small` (`--scale full` streams more Wikipedia; GPU recommended).

**License:** text CC-BY-SA-3.0/4.0 (Wikipedia; attribution: Wikipedia contributors), released
as CC-BY-SA-4.0. Codes and labels: Apache-2.0 (Sphota).

```bibtex
@misc{sphota2026, title={Sphota: An Open, Frozen, Human-Readable Coordinate System for Meaning},
      author={{The Sphota Authors}}, year={2026}}
```
