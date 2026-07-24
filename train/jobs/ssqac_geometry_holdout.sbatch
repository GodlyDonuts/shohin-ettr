#!/bin/bash
#SBATCH -J ssqac-geom
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/ssqac_geometry_%j.out
#SBATCH --open-mode=append
#
# Isolated controller-mechanics fit. This does not load or write the flagship,
# perform language-model training, or authorize continuation pretraining.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=$BASE/miniforge3/bin/python3
RESULT_ROOT=$BASE/artifacts/r12/ssqac_geometry_holdout
OUTPUT=$RESULT_ROOT/report_${SLURM_JOB_ID}.json

SEED=${SEED:-20260724}
TRAIN=${TRAIN:-4096}
EVAL=${EVAL:-512}
WIDTH=${WIDTH:-256}
LAYERS=${LAYERS:-4}
HEADS=${HEADS:-8}
FF=${FF:-1024}
EPOCHS=${EPOCHS:-24}
BATCH=${BATCH:-64}
LR=${LR:-0.0004}
REACTIVE=${REACTIVE:-0}
HIDE_STEP=${HIDE_STEP:-0}

mkdir -p "$BASE/logs" "$RESULT_ROOT"
test ! -e "$OUTPUT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=$BASE:$BASE/train
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[ssqac-geometry] node=${SLURMD_NODENAME:-?} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
EXTRA_ARGS=()
if [[ "$REACTIVE" == "1" ]]; then
  EXTRA_ARGS+=(--reactive)
fi
if [[ "$HIDE_STEP" == "1" ]]; then
  EXTRA_ARGS+=(--hide-step)
fi
"$PYTHON" "$BASE/pipeline/ssqac_controller_trace_pilot.py" \
  --seed "$SEED" \
  --train-examples "$TRAIN" \
  --evaluation-examples "$EVAL" \
  --maximum-rows 6 \
  --maximum-columns 8 \
  --train-maximum-rows 4 \
  --train-maximum-columns 6 \
  --evaluation-minimum-rows 5 \
  --evaluation-minimum-columns 7 \
  --width "$WIDTH" \
  --layers "$LAYERS" \
  --heads "$HEADS" \
  --feedforward "$FF" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --learning-rate "$LR" \
  --maximum-steps 512 \
  --device cuda \
  "${EXTRA_ARGS[@]}" \
  --output "$OUTPUT"

test -s "$OUTPUT"
echo "[ssqac-geometry] report=$OUTPUT"
