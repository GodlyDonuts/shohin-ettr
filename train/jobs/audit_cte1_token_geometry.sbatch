#!/bin/bash
#SBATCH -J cte1-token
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/cte1_token_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
DATA_ROOT=${DATA_ROOT:?set immutable CTE1 data root}
OUT=${OUT:?set isolated output report}
EXPECTED_TRAIN_SHA256=${EXPECTED_TRAIN_SHA256:?set train SHA-256}
EXPECTED_DEVELOPMENT_SHA256=${EXPECTED_DEVELOPMENT_SHA256:?set development SHA-256}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL=$BASE/artifacts/external/qwen3.5-0.8b-2fc0636

test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
export PYTHONPATH="$CODE_ROOT/train:$BASE/envs/idr08-eval-r1/lib/python3.11/site-packages"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
"$PYTHON" "$CODE_ROOT/train/audit_cte1_token_geometry.py" \
  --model-root "$MODEL" \
  --tokenizer-sha256 5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42 \
  --train "$DATA_ROOT/train.jsonl" \
  --expected-train-sha256 "$EXPECTED_TRAIN_SHA256" \
  --development "$DATA_ROOT/development.jsonl" \
  --expected-development-sha256 "$EXPECTED_DEVELOPMENT_SHA256" \
  --output "$OUT"
chmod a-w "$OUT"
