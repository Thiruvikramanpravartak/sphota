#!/usr/bin/env bash
# Full space-v2 pipeline + D-017 decision gate. Designed for one rented GPU (A10 / L4 / A100);
# also runs on CPU at --scale tiny as a dry run.
#
#   bash scripts/gpu_build_v2.sh --scale full            # GPU, ~2-4 h on an A10/L4
#   bash scripts/gpu_build_v2.sh --scale tiny --dry-run  # CPU smoke test (~10-20 min)
#
# Steps: 1) data  2) anchor bake-off (e5-small vs LaBSE vs bge-m3, gate metrics at L1 = 32/64)
#        3) v2 build with the best anchor on a diversified per-language sample
#        4) D-017 PASS/FAIL report (bench/results/v2-gate-<scale>-<anchor>.json)
# Nothing is published. The frozen v1 space is never touched (outputs go to experiments/).
set -euo pipefail
SCALE=full
DRY=0
ANCHORS="intfloat/multilingual-e5-small sentence-transformers/LaBSE BAAI/bge-m3"
SIZES="32 64 256"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scale) SCALE="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --anchors) ANCHORS="$2"; shift 2 ;;
    --sizes) SIZES="$2"; shift 2 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
if [[ $DRY == 1 ]]; then
  # small and fast: one small anchor, one adapter model, tiny data
  ANCHORS="intfloat/multilingual-e5-small"
  ADAPTERS="sentence-transformers/all-MiniLM-L6-v2"
  NADAPT=2000
  SIZES="16 32"
else
  ADAPTERS="sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5 thenlper/gte-small sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 sentence-transformers/LaBSE"
  NADAPT=40000
fi
export SPHOTA_THREADS="${SPHOTA_THREADS:-8}"
echo "== [1/4] data ($SCALE)"
$PY -m sphota.build.data --scale "$SCALE"
echo "== [2-3/4] build + gate per anchor: $ANCHORS"
for A in $ANCHORS; do
  SHORT="$(echo "$A" | tr '/' '_')"
  $PY -m sphota.build.gate v2mini --scale "$SCALE" --anchor "$A" --sizes $SIZES \
      --adapter-models $ADAPTERS --n-adapter "$NADAPT" --out "experiments/v2-$SCALE-$SHORT"
done
echo "== [4/4] summary"
$PY - <<'PYEOF'
import glob, json
for f in sorted(glob.glob("bench/results/v2-gate-*.json")) + sorted(glob.glob("bench/results/v2-mini-gate.json")):
    r = json.load(open(f)); k = r["kept_K1"]
    best = next(x for x in r["results"] if x["K1"] == k)
    flags = " ".join(f"{n.split('_',1)[1]}={'PASS' if v['pass'] else 'FAIL'}" for n, v in best["verdict"].items())
    print(f"{f}: K1={k} {flags}")
PYEOF
