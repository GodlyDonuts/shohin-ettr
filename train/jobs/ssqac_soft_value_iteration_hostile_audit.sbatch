#!/bin/bash
#SBATCH --job-name=ssqac-svi-audit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/ssqac_svi_audit_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/ssqac_svi_audit_%j.err

# Isolated controller audit. It reads no model artifact and writes only its
# causal-feature report.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=$BASE/miniforge3/bin/python3
RESULT_ROOT=$BASE/artifacts/r12/ssqac_soft_value_iteration_hostile
OUTPUT=$RESULT_ROOT/report_${SLURM_JOB_ID}.json

SEED=${SEED:-20260724}
TRAIN=${TRAIN:-96}
EVAL=${EVAL:-64}
EPOCHS=${EPOCHS:-12}
BATCH=${BATCH:-8}
MAX_UPDATES=${MAX_UPDATES:-2048}
WIDTH=${WIDTH:-192}
MESSAGE_LAYERS=${MESSAGE_LAYERS:-3}
ACTION_HIDDEN=${ACTION_HIDDEN:-384}
TRANSITION_HIDDEN=${TRANSITION_HIDDEN:-384}
BACKUPS=${BACKUPS:-8}

mkdir -p "$BASE/logs" "$RESULT_ROOT"
test ! -e "$OUTPUT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=$BASE:$BASE/train
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[ssqac-svi-audit] node=${SLURMD_NODENAME:-?} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
"$PYTHON" "$BASE/train/ssqac_soft_value_iteration_controller.py" \
  --hostile-audit \
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
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --maximum-updates "$MAX_UPDATES" \
  --learning-rate 0.001 \
  --width "$WIDTH" \
  --message-layers "$MESSAGE_LAYERS" \
  --action-hidden "$ACTION_HIDDEN" \
  --transition-hidden "$TRANSITION_HIDDEN" \
  --field-harmonics 4 \
  --coordinate-harmonics 4 \
  --backup-iterations "$BACKUPS" \
  --temperature 0.35 \
  --discount 0.95 \
  --device cuda \
  --output "$OUTPUT"

test -s "$OUTPUT"
sha256sum \
  "$BASE/train/ssqac_soft_value_iteration_controller.py" \
  "$BASE/train/jobs/ssqac_soft_value_iteration_hostile_audit.sbatch" \
  "$OUTPUT"
