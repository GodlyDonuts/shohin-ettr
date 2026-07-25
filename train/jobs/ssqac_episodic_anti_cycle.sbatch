#!/bin/bash
#SBATCH -J ssqac-episodic
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --exclude=evc26,evc31,evc33,evc44
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/ssqac_episodic_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=$BASE/miniforge3/bin/python3
RESULT_ROOT=$BASE/artifacts/r12/ssqac_episodic_anti_cycle

: "${PREPARATION:?PREPARATION must name a sealed CPU artifact}"
SEED=${SEED:-20260724}
UPDATES=${UPDATES:-1500}
BATCH=${BATCH:-2}
LR=${LR:-0.0005}
MAX_STEPS=${MAX_STEPS:-192}

OUTPUT=$RESULT_ROOT/report_${SLURM_JOB_ID}.json
MODEL_DIR=$RESULT_ROOT/models_${SLURM_JOB_ID}

mkdir -p "$BASE/logs" "$RESULT_ROOT"
test -s "$PREPARATION"
test ! -e "$OUTPUT"
test ! -e "$MODEL_DIR"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=$BASE:$BASE/train
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[ssqac-episodic] node=${SLURMD_NODENAME:-?} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[ssqac-episodic] preparation_sha256=$(sha256sum "$PREPARATION" | awk '{print $1}')"
nvidia-smi -L
"$PYTHON" "$BASE/train/ssqac_episodic_anti_cycle_controller.py" run \
  --preparation "$PREPARATION" \
  --seed "$SEED" \
  --optimizer-updates "$UPDATES" \
  --batch-size "$BATCH" \
  --learning-rate "$LR" \
  --maximum-rollout-steps "$MAX_STEPS" \
  --device cuda \
  --output "$OUTPUT" \
  --model-dir "$MODEL_DIR"

test -s "$OUTPUT"
sha256sum "$OUTPUT"
