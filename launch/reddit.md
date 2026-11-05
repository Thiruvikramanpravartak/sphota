# Reddit drafts — ON HOLD, do not post

## r/MachineLearning — [R] Can meaning have universal coordinates? A pre-registered negative-and-positive result

We tried to build a frozen, hierarchical, cross-lingual and cross-model code for meaning
(multilingual-e5-small anchor → meaning-LDA canonical space → 4-level residual k-means) and fixed a
held-out pass/fail gate before the decisive runs: coverage ≥ 60% at L1 confidence ≥ 0.7, confident
translation agreement ≥ 80%, cross-model agreement ≥ 60%, language collapse ≤ 5% per language. We
then amended it to require ≥ 256 top-level cells after a 16-cell dry run trivially "passed".

Results (FLORES-200 devtest, 12 languages):
- No build passes at ≥ 256 cells. Frozen 4096-cell release: 29.4 / 65.1 / 15.7 / 33.8 (0 of 4).
- Granularity cliff: each 4× finer top level costs ~10–25 points; cross-model agreement falls fastest.
  32 cells: 69.0 / 79.3 / 64.5 / 0.0 (3 of 4).
- Language collapse appears abruptly at 256 cells (54.6% of Portuguese test sentences in
  Portuguese-only cells); a diversified sample only reduces it (33.8 → 26.8 at 4096).
- Anchor effect: LaBSE vs e5-small at 32 cells, +10.7 coverage, +4.5 confident translation.
- Positive: hub adapters + stored canonical vectors give 96.5–99.1% recall@10 cross-model without
  re-embedding (codes alone: 10–28%).

Paper, gate definitions, benchmark JSON and CPU-reproducible scripts in the repo. Interested in
whether people working on semantic IDs see the same cliff.

## r/LocalLLaMA — Switch embedding models without re-indexing (and coarse multilingual cache keys), on CPU

Store a 256-d canonical vector per document; an index built with all-MiniLM-L6 can then be queried
with bge-small, gte-small or a multilingual model at 96–99% recall@10, no re-embedding. Coarse
region codes also work as readable, cross-language cache/routing keys. We also tried to make the
codes a universal fine-grained address and it didn't hold up; the write-up says exactly where.
Apache-2.0, ONNX, no torch needed.

## r/LanguageTechnology — Language collapse in multilingual clustering: a cautionary measurement

At fine granularity, k-means over a multilingual encoder (e5-small, even after LDA alignment on
translations) starts forming cells that are ≥ 80% one language. Average cross-lingual agreement
hides this. We propose reporting a per-language collapse metric. Also looking for native speakers
to review machine-generated region labels (issues labelled `labels:<lang>`).
