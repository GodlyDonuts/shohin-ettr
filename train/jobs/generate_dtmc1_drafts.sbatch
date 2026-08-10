#!/bin/bash
#SBATCH -J dtmc1-draft
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/dtmc1_draft_%A_%a.out
#SBATCH --array=0-7
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
OUT_ROOT=${OUT_ROOT:?set isolated output root}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL=$BASE/artifacts/external/qwen3.5-0.8b-2fc0636
CHECKPOINT=$BASE/artifacts/product_reasoning/nmc1/direct_fit_2bad8eb_r1/checkpoint_0001024.pt
DATA=$BASE/artifacts/product_reasoning/nmc1/data_bd30f2a_r1/direct_train.jsonl
OUT=$OUT_ROOT/shard_$(printf '%02d' "$SLURM_ARRAY_TASK_ID").jsonl

test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = \
  8a2b6550f4083368eff2493f1b14d59a1d9dd2e4b39d4da58ff23cdd1b250b53
test "$(sha256sum "$DATA" | awk '{print $1}')" = \
  b78fa53081114f1b5c08f34c0f26df641dfd975a4975130a6b0c8098a0944757

export PYTHONPATH="$CODE_ROOT/train:$BASE/envs/idr08-eval-r1/lib/python3.11/site-packages"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -L
"$PYTHON" "$CODE_ROOT/train/generate_dtmc1_drafts.py" \
  --model-root "$MODEL" \
  --model-revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --checkpoint "$CHECKPOINT" \
  --expected-checkpoint-sha256 8a2b6550f4083368eff2493f1b14d59a1d9dd2e4b39d4da58ff23cdd1b250b53 \
  --data "$DATA" \
  --expected-data-sha256 b78fa53081114f1b5c08f34c0f26df641dfd975a4975130a6b0c8098a0944757 \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --shard-count 8 \
  --batch-size 8 \
  --max-new-tokens 512 \
  --seed 2026081053 \
  --output "$OUT"
chmod a-w "$OUT" "${OUT%.jsonl}.report.json"
