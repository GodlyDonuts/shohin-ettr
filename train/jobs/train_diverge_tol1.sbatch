#!/bin/bash
#SBATCH -J diverge-tol1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/diverge_tol1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
SOURCE_COMMIT=${SOURCE_COMMIT:?set immutable SOURCE_COMMIT}
OUTPUT=${OUTPUT:?set isolated TOL1 output directory}
MODE=${MODE:-component}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
BOARD=${BOARD:-$BASE/artifacts/reasoning/diverge_tol1/board_r1}
FTA1=${FTA1:-$BASE/artifacts/reasoning/diverge_fta1/component_8eeb136_r1/checkpoint_0001600.pt}
FTA1_SHA256=${FTA1_SHA256:-9321b78372d9926930d4de073d70e82c94e8360a69e09be695bab91b2e479f2d}
TRAIN_SHA256=${TRAIN_SHA256:?set frozen TOL1 train hash}
DEVELOPMENT_SHA256=${DEVELOPMENT_SHA256:?set frozen TOL1 development hash}

export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs"
cd "$BASE"

test "$(sha256sum "$BOARD/train.jsonl" | awk '{print $1}')" = "$TRAIN_SHA256"
test "$(sha256sum "$BOARD/development.jsonl" | awk '{print $1}')" = "$DEVELOPMENT_SHA256"
test "$(sha256sum "$FTA1" | awk '{print $1}')" = "$FTA1_SHA256"
test ! -e "$OUTPUT" || { echo "refusing existing output: $OUTPUT" >&2; exit 2; }
trap 'test ! -e "$OUTPUT" || chmod -R a-w "$OUTPUT"' EXIT

echo "[diverge-tol1] node=${SLURMD_NODENAME:-?} code=$CODE_ROOT commit=$SOURCE_COMMIT mode=$MODE"
nvidia-smi -L
"$PYTHON" -m py_compile \
  "$CODE_ROOT/train/diverge_tol1_ir.py" \
  "$CODE_ROOT/train/diverge_tol1_data.py" \
  "$CODE_ROOT/train/diverge_tol1_runtime.py" \
  "$CODE_ROOT/train/diverge_tol1_product.py" \
  "$CODE_ROOT/train/train_diverge_tol1.py"
"$PYTHON" "$CODE_ROOT/train/test_diverge_tol1_ir.py"
"$PYTHON" "$CODE_ROOT/train/test_diverge_tol1_data.py"
"$PYTHON" "$CODE_ROOT/train/test_diverge_tol1_runtime.py"
"$PYTHON" "$CODE_ROOT/train/test_diverge_tol1_product.py"
"$PYTHON" "$CODE_ROOT/pipeline/test_score_diverge_tol1_gate.py"

COMMON=(
  --data "$BOARD/train.jsonl" --data-sha256 "$TRAIN_SHA256"
  --development-data "$BOARD/development.jsonl"
  --development-data-sha256 "$DEVELOPMENT_SHA256"
  --fta1-checkpoint "$FTA1" --fta1-checkpoint-sha256 "$FTA1_SHA256"
  --output "$OUTPUT" --width 192 --layers 2
  --learning-rate 3e-4 --seed 2026080504 --evaluation-batch-size 512
)
if [[ "$MODE" == smoke ]]; then
  "$PYTHON" "$CODE_ROOT/train/train_diverge_tol1.py" "${COMMON[@]}" \
    --updates 2 --batch-size 16 --development-rows 6 --log-interval 1
elif [[ "$MODE" == component ]]; then
  "$PYTHON" "$CODE_ROOT/train/train_diverge_tol1.py" "${COMMON[@]}" \
    --updates 2000 --batch-size 512 --development-rows 512 --log-interval 20
else
  echo "unknown TOL1 mode: $MODE" >&2
  exit 2
fi

echo "[diverge-tol1] completed mode=$MODE"
