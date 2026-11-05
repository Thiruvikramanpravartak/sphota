# Reddit drafts

## r/MachineLearning — [P] Sphota: a frozen, open coordinate system for meaning (codes like `JYZA-7M-X0-A8`)

We built an open "GPS for meaning": a frozen 256-d canonical space (multilingual-e5-small +
meaning-LDA alignment on FLORES/Tatoeba) and a 4-level residual k-means codebook
(4096/256/256/256). Every text gets a hierarchical base32 code with calibrated per-level
confidence.

Things that might interest this sub:
- Alignment *before* quantization: L1 cross-lingual agreement 34% → 46% (12 languages).
- Quantizer bake-off: RQ with EMA + dead-code revival still used fewer effective codes than
  residual k-means; tree k-means degenerates at depth on small data. We report utilization,
  robustness and fidelity separately.
- Calibration by max-likelihood on held-out translations (proper scoring rule); selective
  accuracy tables.
- Hub adapters + canonical vectors: cross-model paraphrase retrieval 96–99% recall@10 without
  re-embedding. Codes alone: 10–28% (coarse) — reported honestly.
- A negative result: whitening against cross-model residuals helped at 1024 cells, not at 4096.

Paper draft, spec, benchmark JSON and everything reproducible on a CPU are in the repo.

## r/LocalLLaMA — Semantic cache & routing that work across languages and embedding models, on CPU

`SemanticCache(depth=2).get_or_compute(question, llm)` hits for "How do I reset my password?",
"¿Cómo restablezco mi contraseña?" and "How can I change my password?" — keys are Sphota code
prefixes, so they're readable and model-portable. Also a prefix router, topic analytics HTML
reports, and a browser demo. Apache-2.0, ONNX, no torch needed.

## r/LanguageTechnology — Multilingual topic codes with labels in 11 languages (incl. Hindi, Tamil, Bengali)

Looking for native speakers to review machine-generated region labels (LLM names + opus-mt
translations). Issues labelled `labels:<lang>` are good first contributions.
