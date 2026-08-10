#!/bin/bash
#SBATCH -J dtmc1-audit
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:20:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/dtmc1_audit_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
TRAIN_DATA=${TRAIN_DATA:?set merged draft corpus}
TRAIN_DATA_SHA256=${TRAIN_DATA_SHA256:?set merged corpus SHA-256}
OUT=${OUT:?set isolated output report}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL=$BASE/artifacts/external/qwen3.5-0.8b-2fc0636
DEV=$BASE/artifacts/product_reasoning/nmc1/data_bd30f2a_r1/development.jsonl
DRAFT_REPORT=$BASE/artifacts/product_reasoning/nmc1/development_f24a237_r2/direct_normal.json

test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
test "$(sha256sum "$TRAIN_DATA" | awk '{print $1}')" = "$TRAIN_DATA_SHA256"
export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline:$BASE/envs/idr08-eval-r1/lib/python3.11/site-packages"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PYTHON" "$CODE_ROOT/pipeline/audit_dtmc1_input_geometry.py" \
  --model-root "$MODEL" \
  --train-data "$TRAIN_DATA" \
  --train-data-sha256 "$TRAIN_DATA_SHA256" \
  --development-data "$DEV" \
  --development-data-sha256 981b83016d9a895af3016b3629c5852b868c80e364dbe97e3901a8d3e8ced4bd \
  --draft-report "$DRAFT_REPORT" \
  --draft-report-sha256 234a029a7479fa056a3bdacd286dcc6e5a1d4cf7b472028b77fec7859ada2a49 \
  --output "$OUT"
chmod a-w "$OUT"
