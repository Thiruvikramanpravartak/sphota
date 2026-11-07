# Space v2 on a GPU (decision run)

**Free route (recommended first): Kaggle.** Upload `notebooks/kaggle_gpu_gate.ipynb` to Kaggle
(*Create → New notebook → File → Import notebook*), turn on a GPU accelerator and Internet, and
*Run all* (the repository is public: do not add any token or secret). It runs the `medium` scale
(12 languages, ~450k sentences) in about 2–4 hours and prints the D-017/D-018 PASS/FAIL table.
Cost ₹0.

One script builds a v2 candidate space and answers the D-017 question (PASS/FAIL per criterion):

```bash
git clone <repo> && cd sphota
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                       # torch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu124
bash scripts/gpu_build_v2.sh --scale full     # ends with the D-017 PASS/FAIL table
```

Dry run on any CPU (tests the whole pipeline, minutes): `bash scripts/gpu_build_v2.sh --scale tiny --dry-run`.

## What it does

1. Downloads data at `full` scale: 20 languages, ~1M English + 200k per language Wikipedia sentences,
   FLORES-200, Tatoeba (`src/sphota/build/common.py`, `SCALES["full"]`).
2. For each anchor (multilingual-e5-small, LaBSE, bge-m3): embeds, fits meaning-LDA on a
   **diversified** sample (≤ 40 sentences per title category, near-duplicates removed, ≤ 8000 per
   language in the clustering corpus) so language-only cells cannot form (D-016), clusters the top
   level at 256 / 1024 / 4096 cells (D-018: the gate is only meaningful with ≥ 256 cells), trains ridge adapters for 5 open models, and evaluates the gate.
3. Writes `bench/results/v2-gate-full-<anchor>.json` and `experiments/v2-full-<anchor>/`
   (experimental artifacts; the frozen v1 space is never modified).

## Hardware, time, cost (estimates)

| GPU | VRAM needed | Embedding speed (bge-m3, fp16) | Full run |
|---|---|---|---|
| NVIDIA L4 (24 GB) | ~6 GB (bge-m3 fp16, batch 64) | ~400–600 sent/s | ~3–4 h |
| NVIDIA A10 (24 GB) | ~6 GB | ~500–800 sent/s | ~2–3 h |
| NVIDIA A100 (40/80 GB) | ~6 GB | ~1500–2500 sent/s | ~1 h |

Dominant cost: embedding ~3.2M sentences with three anchors plus five adapter models on 40k texts.
k-means and the gate are CPU-side and take minutes. RAM: 32 GB recommended; disk: ~40 GB (HF cache +
embeddings). Typical rented-GPU price for this run: a few US dollars to ~US$10.

### Renting a GPU

Any provider with a CUDA image works (Lambda, RunPod, Vast.ai, GCP/AWS/Azure). Pick an Ubuntu +
CUDA 12 image, attach ≥ 50 GB disk, clone the repo, run the commands above inside `tmux`, then copy
back `bench/results/` and `experiments/`. Set `SPHOTA_THREADS` to the vCPU count.

## Decision (D-017)

| # | Criterion | Bar |
|---|---|---|
| 1 | coverage: held-out sentences with L1 confidence ≥ 0.7 | ≥ 60% |
| 2 | translation agreement within the confident portion | ≥ 80% |
| 3 | cross-model L1 agreement (5 adapters) | ≥ 60% |
| 4 | worst language's share in language-only cells | ≤ 5% |
|   | **and** at least 256 top-level cells, all 12 languages (D-018) | |

All four pass → launch as "GPS for meaning" (s2 via RFC + migration table). Otherwise → position as
multilingual topic analytics, routing/caching and vector-based model migration.
