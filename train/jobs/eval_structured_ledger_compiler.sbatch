#!/bin/bash
#SBATCH -J ledger-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/structured_ledger_eval_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
CHECKPOINT=${CHECKPOINT:?set immutable checkpoint}
OUT=${OUT:?set isolated output report}
CONTROL=${CONTROL:?set normal or source_shuffled}
SHARD_COUNT=${SHARD_COUNT:-8}
SHARD_INDEX=${SHARD_INDEX:?set shard index}
LIMIT=${LIMIT:-0}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL=$BASE/artifacts/external/qwen3.5-0.8b-2fc0636
DATA=$BASE/artifacts/product_reasoning/structured_ledger_sft_rg_v4_r1/development.jsonl

test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
test "$(sha256sum "$DATA" | awk '{print $1}')" = \
  760044d9b3851197988b361eba021ffbfe013600fab464731ab72de5e617ec87
export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline:$BASE/envs/idr08-eval-r1/lib/python3.11/site-packages"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -L
"$PYTHON" "$CODE_ROOT/train/eval_structured_ledger_compiler.py" \
  --model-root "$MODEL" \
  --model-revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --checkpoint "$CHECKPOINT" \
  --data "$DATA" \
  --expected-data-sha256 760044d9b3851197988b361eba021ffbfe013600fab464731ab72de5e617ec87 \
  --output "$OUT" \
  --control "$CONTROL" \
  --shard-count "$SHARD_COUNT" \
  --shard-index "$SHARD_INDEX" \
  --limit "$LIMIT" \
  --batch-size 8 \
  --max-new-tokens 320 \
  --seed 2026081023
chmod a-w "$OUT"
