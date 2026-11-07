# FAQ

**Why not just store vectors?**
Vectors are model-private, opaque and heavy. Codes are portable across models (via
adapters), readable, groupable in SQL, tiny (13 characters), and stable forever. Keep vectors
too if you need fine ranking; codes are the shared index.

**Do translations always get the same code?**
No. Discretizing a continuous space cuts some near-identical meanings apart. Agreement is
highest at L1 and falls with depth; confidence tells you how much to trust each level, and
alternates/neighbours handle boundaries. The measured numbers are in [Benchmarks](benchmark.md).

**Why is the anchor such a small model?**
So the reference encoder runs everywhere: Python on a laptop CPU, a phone, and the browser
(the Meaning Atlas encodes in-page). Bigger models (bge-m3, LaBSE) map in through adapters.
We tested bigger anchors on a GPU (D-022): bge-m3 and LaBSE fix language collapse and improve
translation agreement, but codes still do not hold across models at fine resolution, so there is
no `s2` on that evidence.

**Will codes change when I upgrade the library?**
No. Space s1 is frozen; golden tests in CI pin the codes. A future `s2` will ship with a
migration table and the `s2:` tag.

**Can I use OpenAI / Cohere embeddings?**
Yes, with an adapter you train yourself (`sphota train-adapter`, see
`adapters/recipes/`). We do not distribute adapters trained on proprietary API outputs.

**Which languages?**
The anchor covers ~100 languages. The s1 space was aligned and calibrated on 12 (en, hi,
ta, es, fr, de, zh, ar, ru, ja, pt, bn); other languages work but are less tested.

**Is it free?**
Code: Apache-2.0. Release artifacts: Apache-2.0, derived from CC-BY-SA data (attribution in
`NOTICE`). Models keep their own (MIT / Apache-2.0) licenses.
