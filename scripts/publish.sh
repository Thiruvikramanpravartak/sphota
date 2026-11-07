#!/usr/bin/env bash
# Manual publishing from a laptop. Preferred: the GitHub workflows (see RELEASING.md).
#
#   bash scripts/publish.sh pypi        # build + upload the Python package (needs UV_PUBLISH_TOKEN)
#   bash scripts/publish.sh atlas-pages # Meaning Atlas → GitHub Pages
#   bash scripts/publish.sh atlas-hf    # Meaning Atlas → Hugging Face Space (HF_SPACE=<user>/<name>)
#   bash scripts/publish.sh dataset     # tagged Wikipedia sample → HF dataset (HF_DATASET=<user>/<name>)
#   bash scripts/publish.sh all
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
step="${1:-}"
pypi() {
  python -m pytest -q
  rm -rf dist && uv build
  uv publish            # uses UV_PUBLISH_TOKEN (PyPI API token)
}
pages() { bash scripts/deploy_atlas_pages.sh; }
hf() { bash scripts/deploy_atlas_hf.sh; }
dataset() {
  : "${HF_DATASET:?set HF_DATASET=<user>/<name>}"
  python -m sphota.build.open_data --scale small
  python scripts/upload_dataset.py --repo "$HF_DATASET" --scale small
}
case "$step" in
  pypi) pypi ;; atlas-pages) pages ;; atlas-hf) hf ;; dataset) dataset ;;
  all) pypi; pages; hf; dataset ;;
  *) echo "usage: $0 {pypi|atlas-pages|atlas-hf|dataset|all}"; exit 2 ;;
esac
