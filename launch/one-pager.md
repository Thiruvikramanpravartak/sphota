# Sphota — press one-pager

**What:** an open, frozen, human-readable coordinate system for meaning ("GPS for meaning").
Any text in any language → a short hierarchical code (e.g. `JYZA-7M-X0-A8`) with a calibrated
confidence per level and labels in 11 languages.

**Why now:** organizations run search, RAG and analytics on embedding vectors that are tied to
one model. Every model upgrade forces re-embedding entire corpora. Vectors can't be read, shared
or grouped. Sphota provides a stable, shared reference frame; models connect through small
adapters.

**Key facts**
- Frozen space `s1`: 256-d canonical space + 4-level codebook (4096·256·256·256), checksummed.
- 12 languages aligned and calibrated (incl. Hindi, Tamil, Bengali, Arabic, Chinese, Japanese).
- Translations share the top-level code 46% of the time, paraphrases 72%; confident codes are
  more reliable (selective accuracy in the benchmark).
- Cross-model search through the shared space: 96–99% recall@10 without re-indexing.
- Runs on CPU and in the browser; numpy-only core; air-gapped bundles.
- RFC-style specification, conformance certification, Apache-2.0.

**Name:** from Bhartṛhari's *sphoṭa* (Sanskrit philosophy of language): meaning that bursts
forth whole and stays the same across its expressions.

**Links:** repository, Meaning Atlas (live demo), paper draft, benchmark.
**Contact:** maintainers via GitHub.
