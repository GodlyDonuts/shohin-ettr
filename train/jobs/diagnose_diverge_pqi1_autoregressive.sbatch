#!/bin/bash
#SBATCH --job-name=pqi1-ar
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=00:20:00
#SBATCH --exclude=evc33,evc43
#SBATCH --output=/lustre/fs1/home/%u/shohin/logs/pqi1_ar_%j.out

set -euo pipefail

: "${BASE_CODE_ROOT:?}"
: "${OVERLAY_ROOT:?}"
: "${SOURCE_COMMIT:?}"
: "${BASE:?}"
: "${BASE_SHA256:?}"
: "${TOKENIZER:?}"
: "${TOKENIZER_SHA256:?}"
: "${BACKBONE_NAME:?}"
: "${DATA:?}"
: "${DATA_SHA256:?}"
: "${OUTPUT:?}"

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

if [[ -e "$OUTPUT" ]]; then
  echo "refusing existing attribution output: $OUTPUT" >&2
  exit 1
fi

cd "$BASE_CODE_ROOT"
export PYTHONPATH="$BASE_CODE_ROOT/train${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

echo "[pqi1-ar] node=$(hostname) backbone=$BACKBONE_NAME commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=index,name --format=csv,noheader
"$PYTHON" -u "$OVERLAY_ROOT/train/diagnose_diverge_pqi1_autoregressive.py" \
  --base "$BASE" \
  --base-sha256 "$BASE_SHA256" \
  --tokenizer "$TOKENIZER" \
  --tokenizer-sha256 "$TOKENIZER_SHA256" \
  --backbone-name "$BACKBONE_NAME" \
  --data "$DATA" \
  --data-sha256 "$DATA_SHA256" \
  --output "$OUTPUT" \
  --batch-size 64 \
  --device cuda
