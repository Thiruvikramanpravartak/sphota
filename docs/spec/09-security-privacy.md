# Sphota Specification v1, Part 9: Security and Privacy Considerations

## 1. Codes reveal topics

A Sphota code is a *compressed description of what a text is about*. It is designed to be
shared and compared, so it necessarily leaks meaning:

| Depth | What an observer learns (s1) | Example |
|---|---|---|
| L1 | broad topic area (1 of 4096) | "corporate leadership" |
| L2 | sub-topic (≈ 1M cells) | "executive resignation" |
| L3 | narrow theme; near-paraphrase neighbourhoods | "CEO resigns after scandal" |
| L4 | close to a fingerprint of the sentence's meaning | (specific event) |

Codes are **not anonymization** and **not encryption**. In particular:

* **Membership/linkage.** Given a candidate text, anyone can encode it and test whether a
  code (or full code) appears in a shared dataset.
* **Attribute inference.** The L1/L2 labels state topics directly (health, religion,
  politics, sexuality, …), which may be sensitive personal data under GDPR Art. 9.
* **Inversion.** Codes cannot be inverted to the original text, but the reconstruction
  vector supports retrieving *similar* texts from a corpus.

## 2. Privacy by depth

`encode(text, max_level=k)` returns at most `k` levels. Recommended defaults when sharing
outside a trust boundary:

* `max_level=1` for aggregate analytics over personal data,
* `max_level=2` for routing / caching keys,
* full depth only inside the trust boundary (dedupe, search).

Aggregate with k-anonymity thresholds (e.g. suppress prefixes with < 10 items) before
publishing counts. `sphota analyze --min-count` implements suppression.

## 3. Supply chain

* Release and adapter files are `.npz` / `.json` only; no pickles (`allow_pickle=False`).
* SHA-256 checksums of every release file live in `manifest.json` and are verified on load;
  registry adapters carry their own SHA-256.
* Models are fetched from the Hugging Face Hub by repository id; `trust_remote_code` is never
  enabled. Pin revisions in production (`SPHOTA_ANCHOR_REVISION`).
* `SPHOTA_OFFLINE=1` forbids network access; `sphota bundle` produces an air-gapped bundle with
  a checksummed manifest.

## 4. Adversarial inputs

Like any embedding, codes can be steered by adversarial text (keyword stuffing, prompt-like
content). Do not use codes alone for security decisions (moderation, access control).
Confidence is not robustness to adversaries; it is calibrated on natural paraphrase and
translation noise only.

## 5. Bias

The space is derived from Wikipedia, FLORES and Tatoeba in 12 languages; regions are
denser for topics and languages that are well represented there. Labels are keyphrases from
that data and inherit its biases.
