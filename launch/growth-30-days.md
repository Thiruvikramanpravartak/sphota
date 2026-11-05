# 30-day plan — ON HOLD (nothing starts without owner sign-off)

Positioning: a research result first ("Can meaning have universal coordinates? A
negative-and-positive result"), a practical toolkit second (coarse multilingual regions;
model migration via canonical vectors). Not "GPS for meaning".

## Before day 0
- [ ] Owner sign-off on `docs/paper/sphota.md` and these drafts
- [ ] Decide on the one bounded GPU run (bge-m3 anchor, ≥ 256 cells declared in advance); update
      the paper either way
- [ ] README rewritten to match the paper (no universal-coordinate claims)
- [ ] Coarse-region build (32–64 cells) available if the toolkit leads with regions
- [ ] Paper on arXiv (cs.CL, cross-list cs.IR), linked from README
- [ ] 5–10 reviewers from semantic-ID, multilingual NLP and IR lined up for feedback

## Week 1: the result
- [ ] Day 0: paper + X thread + LinkedIn (launch/x-thread.md, launch/linkedin.md)
- [ ] Day 0–1: Show HN (launch/hacker-news.md); stay in the thread
- [ ] Day 1: r/MachineLearning [R], r/LanguageTechnology (launch/reddit.md)
- [ ] Day 3: collect replications / disagreements; triage issues within 24 h

## Week 2: what works
- [ ] Blog: "Change embedding models without re-indexing: canonical vectors as a hub"
- [ ] r/LocalLLaMA post on migration and coarse cache keys
- [ ] Cookbook entries (Qdrant/Chroma/Postgres) for storing canonical vectors

## Weeks 3–4: invite replication
- [ ] Publish the gate as a reusable benchmark script for other encoders / semantic-ID methods
- [ ] Invite results with other anchors (bge-m3, SONAR) under the same pre-declared bar
- [ ] Native-speaker label review drive (Hindi, Tamil, Bengali, Arabic, …)

## Metrics to watch
Replications and citations, benchmark submissions, issues/PRs, migration-feature usage,
GitHub stars and PyPI downloads (secondary).
