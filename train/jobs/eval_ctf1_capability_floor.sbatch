#!/bin/bash
#SBATCH -J ctf1-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/ctf1_eval_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
CONTROL=${CONTROL:?set frozen control}
OUT=${OUT:?set isolated output report}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL=$BASE/artifacts/external/qwen3.5-4b-851bf6e
LAM=$BASE/artifacts/product_reasoning/lam1/full_6b1c1f3_r1/microcode.pt
DATA=$BASE/artifacts/product_reasoning/cte1/data_70d6a76_r4/development.jsonl

test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
test "$(sha256sum "$DATA" | awk '{print $1}')" = \
  aff466172c74dd7d13a183d117e32ec10d5da2048d10253189e4e3b3a599eb04
test "$(sha256sum "$LAM" | awk '{print $1}')" = \
  baab62ec4a256e89950a257616d9f023b8f5e5c8c98b927052f7e722e815c7c4
export PYTHONPATH="$CODE_ROOT/train:$BASE/envs/idr08-eval-r1/lib/python3.11/site-packages"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -L
"$PYTHON" "$CODE_ROOT/train/eval_ctf1_capability_floor.py" \
  --control "$CONTROL" \
  --model-root "$MODEL" \
  --model-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --lam-checkpoint "$LAM" \
  --data "$DATA" \
  --expected-data-sha256 aff466172c74dd7d13a183d117e32ec10d5da2048d10253189e4e3b3a599eb04 \
  --max-new-tokens 512 \
  --batch-size 4 \
  --seed 2026081053 \
  --output "$OUT"
chmod a-w "$OUT"
