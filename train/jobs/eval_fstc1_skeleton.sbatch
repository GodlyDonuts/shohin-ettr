#!/bin/bash
#SBATCH -J fstc1-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/fstc1_eval_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
CHECKPOINT=${CHECKPOINT:?set qualified FSTC1 checkpoint}
OUT=${OUT:?set isolated output report}
CONTROL=${CONTROL:?set normal, source_shuffled, or recurrence_reset}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL=$BASE/artifacts/external/qwen3.5-0.8b-2fc0636
TRAIN=$BASE/artifacts/product_reasoning/structured_ledger_rg_v4_r1/train.jsonl
DATA=$BASE/artifacts/product_reasoning/structured_ledger_rg_v4_r1/development.jsonl

test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
test "$(sha256sum "$TRAIN" | awk '{print $1}')" = \
  02bd92260538ed75c7b1e82aa7cdb0bed4029ed06c8e982ef1b059e5605699df
test "$(sha256sum "$DATA" | awk '{print $1}')" = \
  f5d4a7150c384da046b3e86c76f9b73e1fdc15d4b56add29c3cf0fa83450bdee
export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline:$BASE/envs/idr08-eval-r1/lib/python3.11/site-packages"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -L
"$PYTHON" "$CODE_ROOT/train/eval_fstc1_skeleton.py" \
  --model-root "$MODEL" \
  --model-revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --model-loader auto \
  --checkpoint "$CHECKPOINT" \
  --expected-train-sha256 02bd92260538ed75c7b1e82aa7cdb0bed4029ed06c8e982ef1b059e5605699df \
  --data "$DATA" \
  --expected-data-sha256 f5d4a7150c384da046b3e86c76f9b73e1fdc15d4b56add29c3cf0fa83450bdee \
  --output "$OUT" \
  --control "$CONTROL" \
  --shard-count 1 \
  --shard-index 0 \
  --batch-size 64 \
  --max-source-tokens 256 \
  --shuffle-seed 2026081033
chmod a-w "$OUT"
