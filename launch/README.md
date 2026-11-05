# Launch kit — ON HOLD

**Do not publish any of these drafts.** Nothing goes out until the owner approves (DECISIONS
D-015, D-017, D-018).

## Positioning (changed)
The "GPS for meaning" pitch is retired. The D-017 gate failed on every build we made, and the
D-018 amendment (≥ 256 top-level cells) rules out the 32–64-cell near-passes as evidence for a
universal coordinate system. The drafts now lead with the research result and the parts that work:

* **The finding:** "Can meaning have universal coordinates? A negative-and-positive result"
  (`docs/paper/sphota.md`). Fine-grained universal codes fail a pre-registered held-out gate;
  each 4× finer top level costs 10–25 points; fine partitions collapse into language cells.
* **What works:** coarse shared regions (32–64 cells; confident translations agree 78–89%) for
  multilingual topic analytics, routing and caching; and model migration through stored
  canonical vectors (96.5–99.1% recall@10 without re-embedding).

## Words to avoid
"GPS for meaning", "every sentence has an address", "universal", "frozen forever", "means the
same thing forever", any 4096-cell agreement figure presented as a strength.

## Before un-holding
1. Owner sign-off on the paper.
2. Optional: the one bounded GPU run (bge-m3 anchor, ≥ 256 cells declared in advance). Pass →
   revisit positioning; fail → these drafts stand with updated numbers.
3. Ship a coarse-region (32–64 cell) build if the product leads with regions; the frozen v1
   release is 4096 cells.
