#!/bin/bash
#SBATCH -J cte1-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/cte1_eval_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
CONTROL=${CONTROL:?set frozen control}
CHECKPOINT=${CHECKPOINT:?set immutable checkpoint}
EXPECTED_CHECKPOINT_SHA256=${EXPECTED_CHECKPOINT_SHA256:?set checkpoint SHA-256}
DATA=${DATA:?set immutable CTE1 development data}
EXPECTED_TRAIN_SHA256=${EXPECTED_TRAIN_SHA256:?set training data SHA-256}
EXPECTED_DEVELOPMENT_SHA256=${EXPECTED_DEVELOPMENT_SHA256:?set development data SHA-256}
OUT=${OUT:?set isolated output report}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL=$BASE/artifacts/external/qwen3.5-0.8b-2fc0636
LAM=$BASE/artifacts/product_reasoning/lam1/full_6b1c1f3_r1/microcode.pt

test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$EXPECTED_CHECKPOINT_SHA256"
test "$(sha256sum "$DATA" | awk '{print $1}')" = "$EXPECTED_DEVELOPMENT_SHA256"
test "$(sha256sum "$LAM" | awk '{print $1}')" = \
  baab62ec4a256e89950a257616d9f023b8f5e5c8c98b927052f7e722e815c7c4
export PYTHONPATH="$CODE_ROOT/train:$BASE/envs/idr08-eval-r1/lib/python3.11/site-packages"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -L
"$PYTHON" "$CODE_ROOT/train/eval_cte1_development.py" \
  --control "$CONTROL" \
  --model-root "$MODEL" \
  --model-revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --checkpoint "$CHECKPOINT" \
  --expected-train-sha256 "$EXPECTED_TRAIN_SHA256" \
  --lam-checkpoint "$LAM" \
  --data "$DATA" \
  --expected-data-sha256 "$EXPECTED_DEVELOPMENT_SHA256" \
  --max-new-tokens 512 \
  --batch-size 8 \
  --seed 2026081053 \
  --output "$OUT"
chmod a-w "$OUT"
