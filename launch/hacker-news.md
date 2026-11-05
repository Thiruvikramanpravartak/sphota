# Show HN draft

**Title:** Show HN: Sphota – GPS for meaning: frozen, human-readable codes for text in any language

**URL:** https://github.com/thiruvikramanpravartak/sphota

**First comment (author):**

Hi HN. Sphota maps any text, in any language, to a short hierarchical code like
`JYZA-7M-X0-A8`. The first block is a broad topic region (1 of 4096), each further block zooms
in. The codebook and the 256-d space it lives in are frozen: a code means the same thing forever.

Why: embeddings are great for computation but bad as a data format. They're model-private (a model
upgrade means re-embedding everything), unreadable, and you can't GROUP BY them. I wanted
something like latitude/longitude for meaning.

What's in the repo:
- `pip install sphota` → `encode("The CEO resigned")` → `Sphota('JYZA-7M-X0-A8')`, with a label
  and a per-level confidence. Runs on CPU (ONNX), and in the browser (the Meaning Atlas demo).
- Languages are aligned before quantization (meaning-LDA on FLORES/Tatoeba): +35% relative L1
  agreement vs plain PCA.
- Calibrated confidence per level, so you can ask for "only the levels you're sure about"
  (`min_confidence=0.6` → `JYZA-7M`), plus alternates for near-boundary texts.
- Adapters for other models. If you store the 256-d canonical vector next to the code, an index
  built with all-MiniLM-L6 can be queried with another model at 96–99% recall@10 without
  re-embedding.
- An RFC-style spec, golden tests that pin every code, `sphota certify` for models.

Honest numbers (also in the README): a translation gets the same L1 code 46% of the time, a
paraphrase 72%; deeper levels agree less. Codes alone are a coarse index (great for grouping,
routing, caching, analytics; not for fine ranking). Exact code identity across *different models*
is modest; certification promises neighbourhood-level interoperability. This is a small-scale
CPU build; a GPU full-scale build is next. I also wrote up the negative results.

I'd love feedback on the code format, the spec, and which use cases you'd actually use.
