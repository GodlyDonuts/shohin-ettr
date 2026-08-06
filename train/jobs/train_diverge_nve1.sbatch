#!/bin/bash
#SBATCH -J diverge-nve1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/diverge_nve1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
SOURCE_COMMIT=${SOURCE_COMMIT:?set immutable SOURCE_COMMIT}
OUTPUT=${OUTPUT:?set isolated NVE1 output directory}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
DATA=$BASE/artifacts/reasoning/diverge_nve1/data_da2adcd_r1/evidence_train.jsonl
DATA_SHA256=7eb27276332a14dcbf57c651c8823393fa7eb18b4122cff30c07b961db285b35

export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs"
cd "$BASE"

test "$(sha256sum "$DATA" | awk '{print $1}')" = "$DATA_SHA256"
test ! -e "$OUTPUT" || { echo "refusing existing output: $OUTPUT" >&2; exit 2; }
trap 'test ! -e "$OUTPUT" || chmod -R a-w "$OUTPUT"' EXIT

echo "[diverge-nve1] node=${SLURMD_NODENAME:-?} code=$CODE_ROOT commit=$SOURCE_COMMIT"
nvidia-smi -L
"$PYTHON" -m py_compile \
  "$CODE_ROOT/train/diverge_nve1_data.py" \
  "$CODE_ROOT/train/diverge_nve1_runtime.py" \
  "$CODE_ROOT/train/train_diverge_nve1.py"
"$PYTHON" "$CODE_ROOT/train/test_diverge_nve1_runtime.py"
"$PYTHON" "$CODE_ROOT/train/train_diverge_nve1.py" \
  --data "$DATA" \
  --data-sha256 "$DATA_SHA256" \
  --output "$OUTPUT" \
  --updates 1000 \
  --batch-size 256 \
  --evaluation-batch-size 512 \
  --learning-rate 0.003 \
  --width 192 \
  --layers 2 \
  --seed 2026080610 \
  --log-interval 25

echo "[diverge-nve1] completed"
