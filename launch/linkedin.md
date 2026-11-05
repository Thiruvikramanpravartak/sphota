# LinkedIn post — ON HOLD, do not post

**We tried to give meaning universal coordinates. It didn't work, and that's a useful result.**

Sphota set out to map any text, in any language, to one shared, readable code. We fixed a
pass/fail bar on held-out data before the decisive experiments. No build passed at useful
resolution: every 4× finer level costs 10–25 points of agreement, and fine partitions start
grouping text by language instead of meaning.

What did hold up, and what teams can use:
• Coarse multilingual topic regions for analytics, routing and caching (confident translations
  share a region 79–89% of the time).
• Changing embedding models without re-indexing: store one canonical vector per item and query
  across models at 96–99% recall@10.

The paper, the gate, the negative results and the code are open (Apache-2.0).
github.com/thiruvikramanpravartak/sphota
