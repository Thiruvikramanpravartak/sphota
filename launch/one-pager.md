# Sphota — press one-pager (ON HOLD, do not distribute)

**What:** an open research project and toolkit that asked whether meaning can have universal,
human-readable coordinates across languages and embedding models. The answer, measured against
a gate fixed in advance: **not at fine resolution; yes for coarse regions; and the shared
continuous space is a practical bridge between models.**

**The negative result**
- No build passed the pre-registered held-out gate (coverage, translation agreement, cross-model
  agreement, language collapse) at ≥ 256 top-level cells.
- Each 4× finer top level costs roughly 10–25 points of agreement.
- At fine resolution the clustering starts grouping by *language* instead of meaning (up to 55%
  of Portuguese test sentences in Portuguese-only cells).
- A better anchor helps (LaBSE vs e5-small: +11 pt coverage, +4.5 pt translation agreement) but
  buys about one step of resolution, not a universal code.

**What works**
- **Coarse multilingual regions** (32–64): confident translations share a region 79–89% of the
  time, no language collapse. Useful for topic analytics, routing and semantic caching.
- **Model migration:** store a 256-d canonical vector per item; query an index built with one
  model using another at 96.5–99.1% recall@10, no re-embedding.

**Practical:** numpy-only core, runs on CPU and in the browser, Apache-2.0, all benchmarks and
negative results published.

**Name:** from Bhartṛhari's *sphoṭa*, meaning that stays the same across its expressions; our
results say it does so only at a coarse resolution.

**Links:** repository, paper, benchmark, gate definitions (DECISIONS D-017/D-018).
