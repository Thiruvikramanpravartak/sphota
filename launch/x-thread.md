# X / Twitter thread — ON HOLD, do not post

1/ Can meaning have universal coordinates, a code every sentence gets in any language, from any
model? We tried, with a pass/fail bar fixed in advance. Short answer: not at fine resolution.
Paper + all numbers 🧵

2/ The gate (held-out, 12 languages): coverage ≥ 60%, confident translation agreement ≥ 80%,
cross-model ≥ 60%, no language collapse. Must hold with ≥ 256 top-level cells. Every build failed.

3/ The trade-off is brutal: each 4× finer top level costs 10–25 points. 32 cells: 69/79/65/0 (almost
passes). 4096 cells: 29/65/16/34. Agreement is bought with coarseness.

4/ At fine resolution, clustering starts grouping by language instead of meaning: 55% of Portuguese
test sentences in Portuguese-only cells. Watch for this in any multilingual quantization.

5/ Better anchor = about one step of resolution. LaBSE vs e5-small at 32 cells: +11 pt coverage,
+4.5 pt translation agreement. Helpful, not universal.

6/ What works: coarse shared regions for multilingual analytics/routing/caching, and model
migration via stored canonical vectors: 96–99% recall@10 across models, no re-embedding.

7/ Everything is open: paper, gate, negative results, code. github.com/thiruvikramanpravartak/sphota
