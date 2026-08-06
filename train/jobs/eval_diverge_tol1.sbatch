#!/bin/bash
#SBATCH -J eval-tol1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/eval_diverge_tol1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
CHECKPOINT=${CHECKPOINT:?set immutable TOL1 checkpoint}
CHECKPOINT_SHA256=${CHECKPOINT_SHA256:?set immutable TOL1 checkpoint hash}
OUTPUT=${OUTPUT:?set isolated TOL1 evaluation path}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
BOARD=${BOARD:-$BASE/artifacts/reasoning/diverge_tol1/board_r1}
OOD_SHA256=${OOD_SHA256:?set frozen TOL1 OOD hash}

export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs" "$(dirname "$OUTPUT")"
cd "$BASE"

test "$(sha256sum "$BOARD/ood.jsonl" | awk '{print $1}')" = "$OOD_SHA256"
test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$CHECKPOINT_SHA256"
test ! -e "$OUTPUT" || { echo "refusing existing evaluation: $OUTPUT" >&2; exit 2; }

"$PYTHON" "$CODE_ROOT/train/eval_diverge_tol1.py" \
  --checkpoint "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --data "$BOARD/ood.jsonl" --data-sha256 "$OOD_SHA256" \
  --output "$OUTPUT" --batch-size 512
"$PYTHON" "$CODE_ROOT/pipeline/score_diverge_tol1_gate.py" \
  --evaluation "$OUTPUT" --output "${OUTPUT%.json}.gate.json"
