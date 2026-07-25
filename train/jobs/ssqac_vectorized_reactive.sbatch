#!/bin/bash
#SBATCH --job-name=ssqac-vectorized
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/ssqac_vectorized_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/ssqac_vectorized_%j.err

# Isolated controller-mechanics fit. This script cannot load or write the
# flagship checkpoint and does not authorize continuation pretraining.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=$BASE/miniforge3/bin/python3
RESULT_ROOT=$BASE/artifacts/r12/ssqac_vectorized_reactive
OUTPUT=$RESULT_ROOT/report_${SLURM_JOB_ID}.json
MODEL=$RESULT_ROOT/model_${SLURM_JOB_ID}.pt

SEED=${SEED:-20260724}
TRAIN=${TRAIN:-8192}
AUDIT=${AUDIT:-1024}
EVAL=${EVAL:-512}
WIDTH=${WIDTH:-256}
BLOCKS=${BLOCKS:-6}
FF=${FF:-768}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-1024}
LR=${LR:-0.0003}

mkdir -p "$BASE/logs" "$RESULT_ROOT"
test ! -e "$OUTPUT"
test ! -e "$MODEL"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=$BASE:$BASE/train
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[ssqac-vectorized] node=${SLURMD_NODENAME:-?} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
"$PYTHON" "$BASE/pipeline/ssqac_vectorized_reactive_pilot.py" \
  --seed "$SEED" \
  --train-examples "$TRAIN" \
  --audit-examples "$AUDIT" \
  --evaluation-examples "$EVAL" \
  --fit-maximum-rows 4 \
  --fit-maximum-columns 6 \
  --evaluation-minimum-rows 5 \
  --evaluation-minimum-columns 7 \
  --maximum-rows 6 \
  --maximum-columns 8 \
  --registers 4 \
  --width "$WIDTH" \
  --blocks "$BLOCKS" \
  --feedforward "$FF" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --learning-rate "$LR" \
  --weight-decay 0.01 \
  --maximum-rollout-steps 512 \
  --device cuda \
  --amp-bfloat16 \
  --compile \
  --material-minimum-evaluation-cases 512 \
  --material-minimum-certification-rate 0.8 \
  --model-output "$MODEL" \
  --output "$OUTPUT"

test -s "$OUTPUT"
test -s "$MODEL"
echo "[ssqac-vectorized] report=$OUTPUT model=$MODEL"
