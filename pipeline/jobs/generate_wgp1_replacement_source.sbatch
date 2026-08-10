#!/bin/bash
#SBATCH -J wgp1-source
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/wgp1_source_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
OUT=${OUT:?set isolated output root}
VENDOR=${VENDOR:?set pinned reasoning-gym vendor root}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
PROTECTED=$BASE/artifacts/product_reasoning/btt1_byte_supervision_cf4f9cb_r1

test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
test "$(sha256sum "$BASE/vendor/wheels_wgp1/reasoning_gym-0.1.25-py3-none-any.whl" | awk '{print $1}')" = \
  7f17a3eddb13c015d7d4a755ed576a061df889faf9468bcc2cca334ebe9e0435
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$VENDOR:$CODE_ROOT/pipeline"
"$PYTHON" "$CODE_ROOT/pipeline/generate_wgp1_replacement_source.py" \
  --protected "$PROTECTED/train.jsonl" \
  --protected "$PROTECTED/development.jsonl" \
  --output-root "$OUT"
find "$OUT" -type f -exec chmod a-w {} +
