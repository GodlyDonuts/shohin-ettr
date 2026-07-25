#!/bin/bash
#SBATCH -J ssqac-successor
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --exclude=evc31,evc33,evc36,evc44
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/ssqac_successor_%j.out
#SBATCH --open-mode=append
#
# Isolated successor-planning falsifier. It does not load or write a language
# model checkpoint, run pretraining, or touch any EFC/SVI/canonical artifact.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=$BASE/miniforge3/bin/python3
RESULT_ROOT=$BASE/artifacts/r12/ssqac_successor_value
OUTPUT=$RESULT_ROOT/report_${SLURM_JOB_ID}.json
MODEL_DIR=$RESULT_ROOT/models_${SLURM_JOB_ID}

SEED=${SEED:-20260724}
TRAIN=${TRAIN:-128}
EVAL=${EVAL:-96}
UPDATES=${UPDATES:-1200}
BATCH=${BATCH:-4}
LR=${LR:-0.0006}
FIELD_WIDTH=${FIELD_WIDTH:-64}
WIDTH=${WIDTH:-256}
CELL_HIDDEN=${CELL_HIDDEN:-384}
MATRIX_LAYERS=${MATRIX_LAYERS:-4}
PLANNER_HIDDEN=${PLANNER_HIDDEN:-512}
PLANNER_ITERATIONS=${PLANNER_ITERATIONS:-8}

mkdir -p "$BASE/logs" "$RESULT_ROOT"
test ! -e "$OUTPUT"
test ! -e "$MODEL_DIR"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=$BASE:$BASE/train
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[ssqac-successor] node=${SLURMD_NODENAME:-?} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
"$PYTHON" "$BASE/train/ssqac_successor_value_controller.py" \
  --seed "$SEED" \
  --train-matrices "$TRAIN" \
  --evaluation-matrices "$EVAL" \
  --train-maximum-rows 3 \
  --train-maximum-columns 4 \
  --evaluation-minimum-rows 4 \
  --evaluation-minimum-columns 5 \
  --evaluation-maximum-rows 4 \
  --evaluation-maximum-columns 6 \
  --maximum-preparation-steps 96 \
  --maximum-rollout-steps 192 \
  --optimizer-updates "$UPDATES" \
  --batch-size "$BATCH" \
  --learning-rate "$LR" \
  --field-width "$FIELD_WIDTH" \
  --width "$WIDTH" \
  --cell-hidden "$CELL_HIDDEN" \
  --matrix-layers "$MATRIX_LAYERS" \
  --planner-hidden "$PLANNER_HIDDEN" \
  --planner-iterations "$PLANNER_ITERATIONS" \
  --device cuda \
  --model-dir "$MODEL_DIR" \
  --output "$OUTPUT"

test -s "$OUTPUT"
echo "[ssqac-successor] report=$OUTPUT"
