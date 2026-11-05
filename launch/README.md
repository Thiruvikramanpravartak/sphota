# Launch kit — ON HOLD

Do not use these drafts yet (DECISIONS D-015). They pitch "GPS for meaning", which the
`small`-scale benchmark does not support: only 26% of held-out sentences get a confident (≥ 0.7)
top-level code, and a confident English code is matched by its translation 67% of the time.

Decision gate: run the full-scale GPU experiment with the bar set in advance (L1 translation
agreement ≥ 80% and cross-model ≥ 60% on the confident portion, with adequate coverage).
Pass → update numbers and launch with this kit. Fail → rewrite around multilingual topic
analytics, routing/caching and vector-based model migration.
