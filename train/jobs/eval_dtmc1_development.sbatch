#!/bin/bash
#SBATCH -J dtmc1-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/dtmc1_eval_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
CONTROL=${CONTROL:?set frozen control}
COMPILER=${COMPILER:?set immutable compiler checkpoint}
EXPECTED_COMPILER_SHA256=${EXPECTED_COMPILER_SHA256:?set compiler SHA-256}
EXPECTED_TRAIN_DATA_SHA256=${EXPECTED_TRAIN_DATA_SHA256:?set training data SHA-256}
OUT=${OUT:?set isolated output report}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL=$BASE/artifacts/external/qwen3.5-0.8b-2fc0636
OWNER=$BASE/artifacts/product_reasoning/nmc1/direct_fit_2bad8eb_r1/checkpoint_0001024.pt
DATA=$BASE/artifacts/product_reasoning/nmc1/data_bd30f2a_r1/development.jsonl
DRAFT_REPORT=$BASE/artifacts/product_reasoning/nmc1/development_f24a237_r2/direct_normal.json
LAM=$BASE/artifacts/product_reasoning/lam1/full_6b1c1f3_r1/microcode.pt

test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
test "$(sha256sum "$COMPILER" | awk '{print $1}')" = "$EXPECTED_COMPILER_SHA256"
test "$(sha256sum "$OWNER" | awk '{print $1}')" = \
  8a2b6550f4083368eff2493f1b14d59a1d9dd2e4b39d4da58ff23cdd1b250b53
test "$(sha256sum "$DATA" | awk '{print $1}')" = \
  981b83016d9a895af3016b3629c5852b868c80e364dbe97e3901a8d3e8ced4bd
test "$(sha256sum "$DRAFT_REPORT" | awk '{print $1}')" = \
  234a029a7479fa056a3bdacd286dcc6e5a1d4cf7b472028b77fec7859ada2a49
test "$(sha256sum "$LAM" | awk '{print $1}')" = \
  baab62ec4a256e89950a257616d9f023b8f5e5c8c98b927052f7e722e815c7c4

export PYTHONPATH="$CODE_ROOT/train:$BASE/envs/idr08-eval-r1/lib/python3.11/site-packages"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -L
"$PYTHON" "$CODE_ROOT/train/eval_dtmc1_development.py" \
  --control "$CONTROL" \
  --model-root "$MODEL" \
  --model-revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --owner-checkpoint "$OWNER" \
  --expected-owner-sha256 8a2b6550f4083368eff2493f1b14d59a1d9dd2e4b39d4da58ff23cdd1b250b53 \
  --expected-owner-data-sha256 b78fa53081114f1b5c08f34c0f26df641dfd975a4975130a6b0c8098a0944757 \
  --compiler-checkpoint "$COMPILER" \
  --expected-compiler-sha256 "$EXPECTED_COMPILER_SHA256" \
  --expected-train-data-sha256 "$EXPECTED_TRAIN_DATA_SHA256" \
  --lam-checkpoint "$LAM" \
  --data "$DATA" \
  --expected-data-sha256 981b83016d9a895af3016b3629c5852b868c80e364dbe97e3901a8d3e8ced4bd \
  --draft-report "$DRAFT_REPORT" \
  --expected-draft-report-sha256 234a029a7479fa056a3bdacd286dcc6e5a1d4cf7b472028b77fec7859ada2a49 \
  --batch-size 32 \
  --output "$OUT"
chmod a-w "$OUT"
