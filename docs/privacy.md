# Privacy

!!! warning "Codes reveal topics"
    A Sphota code is a compressed statement of what a text is about. It is **not**
    anonymization and **not** encryption.

| Depth | What it reveals | Recommended for |
|---|---|---|
| L1 (`JYZA`) | broad topic (1 of 4096) | aggregate analytics on personal data |
| L2 (`JYZA-7M`) | sub-topic | routing keys, cache keys |
| L3 | narrow theme | inside your trust boundary |
| L4 | close to a meaning fingerprint of the sentence | inside your trust boundary |

* `encode(text, max_level=k)` never computes or returns more than `k` levels.
* `sphota analyze --min-count 10` suppresses small groups and hides sample texts.
* Anyone can encode a guessed text and check whether its code appears in a shared dataset
  (membership inference). Treat full codes of personal data as personal data.
* Labels name topics directly; some topics are sensitive categories (health, religion,
  politics, sexuality). Apply the same care as to the underlying text.

Full discussion: [Specification, Part 9](spec/09-security-privacy.md).
