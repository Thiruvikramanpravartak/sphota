# Show HN draft — ON HOLD, do not post

**Title:** Show HN: We tried to give meaning universal coordinates. Here's where it broke

**URL:** https://github.com/thiruvikramanpravartak/sphota

**First comment (author):**

Hi HN. I set out to build "GPS for meaning": a frozen, readable code (like `JYZA-7M-X0-A8`) that
any sentence in any language would get, the same whichever embedding model you used. Before the
decisive experiments I wrote down a pass/fail gate on held-out data: ≥ 60% of sentences get a
confident top-level code, ≥ 80% of confident translations agree, ≥ 60% agreement across models,
and no language may collapse into its own cells. Later I added: it must hold with ≥ 256
top-level cells, because a 16-cell dry run "passed" and that proves nothing.

Nothing passed. The frozen 4096-cell release: 29% coverage, 65% translation agreement, 16%
cross-model. Re-clustering shows the trade-off cleanly: every 4× finer top level costs 10–25
points. At 32 cells it nearly passes (69% / 79% / 65% / 0%), but 32 regions is a topic
taxonomy, not coordinates. Swapping the anchor to LaBSE helps (+11 pt coverage, +4.5 pt
translation agreement), about one step of resolution.

The weirdest finding: at ≥ 256 cells k-means starts building *language* cells. Over half of
Portuguese test sentences landed in Portuguese-only cells regardless of topic. A diversified
sample only partly fixed it; part of it is the encoder.

What did work:
- Coarse shared regions (32–64) for multilingual analytics, routing and caching keys.
- Migration: store a 256-d canonical vector next to each item and an index built with
  all-MiniLM-L6 can be queried with another model at 96–99% recall@10 without re-embedding.

Paper (with all the tables), the gate, and scripts to reproduce are in the repo. I'd especially
like to hear from people working on semantic IDs: do you see the same granularity cliff?
