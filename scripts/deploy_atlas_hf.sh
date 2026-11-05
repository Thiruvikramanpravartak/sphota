#!/usr/bin/env bash
# Deploy the built Meaning Atlas to a static Hugging Face Space. NOT run automatically.
#   huggingface-cli login
#   HF_SPACE=<user>/sphota-atlas bash scripts/deploy_atlas_hf.sh
set -euo pipefail
: "${HF_SPACE:?set HF_SPACE=<user>/<space>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
(cd "$ROOT/atlas" && npm ci && npm run build)
TMP="$(mktemp -d)"
cp -r "$ROOT/atlas/dist/." "$TMP/"
cp "$ROOT/hf_space/README.md" "$TMP/README.md"
huggingface-cli repo create "${HF_SPACE#*/}" --type space --space_sdk static -y || true
huggingface-cli upload "$HF_SPACE" "$TMP" . --repo-type space
echo "deployed: https://huggingface.co/spaces/$HF_SPACE"
