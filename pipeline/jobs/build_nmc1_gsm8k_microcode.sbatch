#!/bin/bash
#SBATCH -J nmc1-data
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/nmc1_data_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
OUT=${OUT:?set isolated output root}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
"$PYTHON" "$CODE_ROOT/pipeline/build_nmc1_gsm8k_microcode.py" \
  --train "$BASE/artifacts/problems/gsm8k_train.jsonl" \
  --test "$BASE/artifacts/evals/gsm8k.jsonl" \
  --expected-train-sha256 e0a44460964649db791a0ced18449b34c34875218b7d21b23241cbdb6ca2a104 \
  --expected-test-sha256 752adc99f23132275139f1f7be57126b08735a55daf0c56859ee6df70abafbac \
  --development-modulus 10 \
  --output "$OUT"
find "$OUT" -type f -exec chmod a-w {} +
