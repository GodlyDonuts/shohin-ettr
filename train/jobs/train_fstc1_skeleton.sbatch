#!/bin/bash
#SBATCH -J fstc1-skeleton
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/fstc1_skeleton_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
OUT=${OUT:?set isolated output root}
UPDATES=${UPDATES:-1024}
BATCH_SIZE=${BATCH_SIZE:-32}
GOLD_FEEDBACK_UPDATES=${GOLD_FEEDBACK_UPDATES:-128}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL=$BASE/artifacts/external/qwen3.5-0.8b-2fc0636
DATA=$BASE/artifacts/product_reasoning/structured_ledger_rg_v4_r1/train.jsonl

test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
test "$(sha256sum "$DATA" | awk '{print $1}')" = \
  02bd92260538ed75c7b1e82aa7cdb0bed4029ed06c8e982ef1b059e5605699df
test "$(sha256sum "$MODEL/tokenizer.json" | awk '{print $1}')" = \
  5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42
for path in "$CODE_ROOT/train/fixed_slot_typed_compiler.py" \
            "$CODE_ROOT/train/train_fstc1_skeleton.py"; do
  test -f "$path"
done
export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline:$BASE/envs/idr08-eval-r1/lib/python3.11/site-packages"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -L
"$PYTHON" "$CODE_ROOT/train/train_fstc1_skeleton.py" \
  --model-root "$MODEL" \
  --model-revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --model-loader auto \
  --data "$DATA" \
  --expected-data-sha256 02bd92260538ed75c7b1e82aa7cdb0bed4029ed06c8e982ef1b059e5605699df \
  --output "$OUT" \
  --updates "$UPDATES" \
  --batch-size "$BATCH_SIZE" \
  --gold-feedback-updates "$GOLD_FEEDBACK_UPDATES" \
  --max-source-tokens 256 \
  --width 512 \
  --encoder-layers 4 \
  --heads 8 \
  --learning-rate 2e-4 \
  --seed 2026081031 \
  --data-seed 2026081032 \
  --log-interval 10
find "$OUT" -type f -exec chmod a-w {} +
