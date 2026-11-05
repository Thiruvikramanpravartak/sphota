# Concepts

## Sphota: the meaning that bursts forth

In Bhartṛhari's philosophy of language (*Vākyapadīya*, 5th century CE), **sphoṭa** (स्फोट,
"bursting forth") is the meaning that flashes in the mind as a whole, the same however it is
uttered: in any voice, any accent, any wording. Sounds and letters are only the vehicle.

Sphota the library takes the idea literally: the *vehicle* (language, wording, embedding
model) varies; the *coordinate* of the meaning should not.

## A coordinate system, not a model

```
text (any language) ──► embedding model ──► canonical space (frozen) ──► code JYZA-7M-X0-A8
other model ─────────► adapter ──────────┘
```

* **Canonical space** (s1): 256 dimensions, defined once from an *anchor* model
  (multilingual-e5-small) with a fixed projection that makes translations of the same
  sentence land close together. Frozen forever.
* **Codebook**: four levels of centroids (4096 · 256 · 256 · 256). A text's code is the path of
  nearest centroids, written in Crockford base32.
* **Adapters**: small linear maps from other embedding models into the canonical space. Store
  codes once; switch models later without re-indexing.

## Levels and prefixes

| Level | Example | Meaning |
|---|---|---|
| L1 | `JYZA` | broad topic area (4096 regions) |
| L2 | `JYZA-7M` | sub-topic |
| L3 | `JYZA-7M-X0` | narrow theme |
| L4 | `JYZA-7M-X0-A8` | near-paraphrase neighbourhood |

A prefix is a larger region containing all its extensions. `GROUP BY` a prefix to count topics;
`WHERE code LIKE 'JYZA-7M%'` to filter a region.

## Confidence and precision (GPS accuracy)

Every level carries a calibrated **confidence**: the probability that another expression of
the same meaning (a translation, a paraphrase, another certified model) gets the same prefix.
It decreases with depth. `min_confidence` truncates the code where it gets unsure, like a GPS
fix that reports a wider area instead of a wrong street.

## Boundaries, alternates and stable mode

Any partition of a continuous space cuts some near-identical meanings apart. Sphota is honest
about it:

* `c.alternates`: other likely codes, with probabilities, for near-boundary texts;
* `c.candidates(k)`: the k most probable full codes (index ambiguous items under several);
* `encode(..., stable=True)`: a deterministic "majority vote under small perturbations" that
  prefers the cell most robust to noise (small measured gain; see Benchmarks);
* hierarchy-aware `similarity()` that treats neighbouring regions as close.

## Labels

Every L1 region (4096) and every populated L2 region (~3.8k) has a short human-readable label
in English and 10 more languages (Hindi, Tamil, Bengali, Spanish, French, German, Portuguese,
Chinese, Arabic, Russian; L2 labels in hi/ta/bn/es/fr/zh). Labels are descriptive aids generated
offline: a small open LLM (Qwen2.5) names each region from member sentences and MarianMT
translates. Quality is good for European languages and Chinese, mixed for Hindi, Tamil and
Bengali; Japanese falls back to English (no usable open translation model was available on
CPU). Native-language keyphrases are kept alongside in `keywords.json`.

## What Sphota is not

* Not a replacement for vectors when you need fine-grained ranking: use codes to *organize,
  filter, route, cache and share*; keep vectors (if you have them) for final re-ranking.
* Not anonymization: codes reveal topics. See [Privacy](privacy.md).
