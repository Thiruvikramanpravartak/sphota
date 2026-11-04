---
title: Sphota Meaning Atlas
emoji: 🧭
colorFrom: indigo
colorTo: pink
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: GPS for meaning. Encode any sentence in your browser.
---

# Sphota · Meaning Atlas

A zoomable map of the Sphota s1 meaning space. Type a sentence in any language: it is encoded
**in your browser** (multilingual-e5-small via transformers.js) and lands on its code.

Built from `atlas/` with `npm run build`; deployed with `scripts/deploy_atlas_hf.sh`.
A Python (Gradio) demo is in `app.py` for a `sdk: gradio` Space.
