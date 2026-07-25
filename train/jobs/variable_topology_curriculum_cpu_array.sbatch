#!/bin/bash
#SBATCH -J variable-type-curriculum
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=20G
#SBATCH -t 02:00:00
#SBATCH --array=0-14%15
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/variable_type_curriculum_%A_%a.out
#SBATCH --open-mode=append
#SBATCH --exclude=evc26,evc31,evc33,evc44

set -euo pipefail

ROOT="${ROOT:-/lustre/fs1/home/sa305415/shohin}"
PYTHON="$ROOT/miniforge3/bin/python3"
FAMILIES=(affine_modular bitwise_rotate_xor permutation)
SEED_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
FAMILY_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
MODEL_SEED=$((20260725 + SEED_INDEX))
HELD_OUT_FAMILY="${FAMILIES[$FAMILY_INDEX]}"
OUTPUT="artifacts/r12/source_deleted_variable_topology_board_v1/semantic_type_holdout_${HELD_OUT_FAMILY}_seed${MODEL_SEED}.json"

cd "$ROOT"
mkdir -p "$(dirname "$OUTPUT")" logs
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT:$ROOT/train"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "[variable-type-curriculum] task=$SLURM_ARRAY_TASK_ID seed=$MODEL_SEED held_out=$HELD_OUT_FAMILY node=${SLURMD_NODENAME:-?}"
"$PYTHON" train/run_variable_topology_renderer_curriculum.py \
  --seed "$MODEL_SEED" \
  --steps 300 \
  --width 96 \
  --layers 2 \
  --learning-rate 0.003 \
  --device cpu \
  --held-out-family "$HELD_OUT_FAMILY" \
  --train-per-renderer 4 \
  --development-per-cell 4 \
  --output "$OUTPUT"
