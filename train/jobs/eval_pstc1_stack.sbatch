#!/bin/bash
#SBATCH -J pstc1-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/pstc1_eval_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
CHECKPOINT=${CHECKPOINT:?set PSTC1 checkpoint}
OUT=${OUT:?set isolated output report}
CONTROL=${CONTROL:?set normal, source_shuffled, stack_reset, or stack_top_permuted}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL=$BASE/artifacts/external/qwen3.5-0.8b-2fc0636
TRAIN=$BASE/artifacts/product_reasoning/pstc1_stack_supervision_ca5abd6_r1/train.jsonl
DATA=$BASE/artifacts/product_reasoning/pstc1_stack_supervision_ca5abd6_r1/development.jsonl

test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
test "$(sha256sum "$TRAIN" | awk '{print $1}')" = \
  b522d58b3403064987a47d7b22657ef862945f18dfb352a78625fbcc27ebf008
test "$(sha256sum "$DATA" | awk '{print $1}')" = \
  bf90044cd17e5e63cdeb7f91bffaae6e4bc59e992fb95cb8da9d2b29da00c914
export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline:$BASE/envs/idr08-eval-r1/lib/python3.11/site-packages"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -L
"$PYTHON" "$CODE_ROOT/train/eval_pstc1_stack.py" \
  --model-root "$MODEL" \
  --model-revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --model-loader auto \
  --checkpoint "$CHECKPOINT" \
  --expected-train-sha256 b522d58b3403064987a47d7b22657ef862945f18dfb352a78625fbcc27ebf008 \
  --data "$DATA" \
  --expected-data-sha256 bf90044cd17e5e63cdeb7f91bffaae6e4bc59e992fb95cb8da9d2b29da00c914 \
  --output "$OUT" \
  --control "$CONTROL" \
  --batch-size 64 \
  --max-source-tokens 256 \
  --shuffle-seed 2026081043
chmod a-w "$OUT"
