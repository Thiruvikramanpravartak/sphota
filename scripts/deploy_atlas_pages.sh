#!/usr/bin/env bash
# Deploy the Meaning Atlas to GitHub Pages (gh-pages branch). NOT run automatically.
#   bash scripts/deploy_atlas_pages.sh            # serves at https://<user>.github.io/sphota/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/atlas"
npm ci
ATLAS_BASE=/sphota/ npm run build
npx --yes gh-pages@6.3.0 -d dist -m "Deploy Meaning Atlas"
