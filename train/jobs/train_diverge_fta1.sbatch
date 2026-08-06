#!/bin/bash
#SBATCH -J diverge-fta1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/diverge_fta1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
SOURCE_COMMIT=${SOURCE_COMMIT:?set immutable SOURCE_COMMIT}
OUTPUT=${OUTPUT:?set isolated FTA1 output directory}
MODE=${MODE:-component}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
DATA=$BASE/artifacts/reasoning/diverge_crp1/train_seed2026080604.jsonl
DEVELOPMENT_DATA=$BASE/artifacts/reasoning/diverge_crp1/development_seed2026080604.jsonl
DATA_SHA256=33dacc5e0f9a72ad01eaa58bbee627c534437db95752b7e6a76a8488af0b2ede
DEVELOPMENT_DATA_SHA256=ad27c4ca8f89a158fc1c4516e1cfce67f792bd5015f0b68e9c30ba39b62cfde5

export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs"
cd "$BASE"

test "$(sha256sum "$DATA" | awk '{print $1}')" = "$DATA_SHA256"
test "$(sha256sum "$DEVELOPMENT_DATA" | awk '{print $1}')" = "$DEVELOPMENT_DATA_SHA256"
test ! -e "$OUTPUT" || { echo "refusing existing output: $OUTPUT" >&2; exit 2; }
trap 'test ! -e "$OUTPUT" || chmod -R a-w "$OUTPUT"' EXIT

echo "[diverge-fta1] node=${SLURMD_NODENAME:-?} code=$CODE_ROOT commit=$SOURCE_COMMIT mode=$MODE"
nvidia-smi -L
"$PYTHON" -m py_compile \
  "$CODE_ROOT/train/diverge_fta1_runtime.py" \
  "$CODE_ROOT/train/train_diverge_fta1.py"
"$PYTHON" "$CODE_ROOT/train/test_diverge_ats1_data.py"
"$PYTHON" "$CODE_ROOT/train/test_diverge_ats1_runtime.py"
"$PYTHON" "$CODE_ROOT/train/test_diverge_fta1_runtime.py"
"$PYTHON" "$CODE_ROOT/pipeline/test_score_diverge_fta1_component.py"

COMMON=(
  --data "$DATA" --data-sha256 "$DATA_SHA256"
  --development-data "$DEVELOPMENT_DATA"
  --development-data-sha256 "$DEVELOPMENT_DATA_SHA256"
  --output "$OUTPUT" --width 192 --layers 2
  --learning-rate 3e-4 --seed 2026080607 --evaluation-batch-size 1024
)
if [[ "$MODE" == smoke ]]; then
  "$PYTHON" "$CODE_ROOT/train/train_diverge_fta1.py" "${COMMON[@]}" \
    --updates 2 --batch-size 16 --development-rows 6 --log-interval 1
elif [[ "$MODE" == component ]]; then
  "$PYTHON" "$CODE_ROOT/train/train_diverge_fta1.py" "${COMMON[@]}" \
    --updates 1600 --batch-size 512 --development-rows 96 --log-interval 20
else
  echo "unknown FTA1 mode: $MODE" >&2
  exit 2
fi

echo "[diverge-fta1] completed mode=$MODE"
