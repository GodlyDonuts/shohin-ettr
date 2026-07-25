#!/bin/bash
#SBATCH -J ssqac-proof
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --exclude=evc31,evc33,evc44
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/ssqac_proof_%j.out
#SBATCH --open-mode=append
#
# GPU-only fit/eval phase for the isolated proof-carrying action falsifier.
# PREPARATION must name a sealed CPU-generated artifact.  This job never loads
# or writes a language-model checkpoint and cannot start pretraining.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=$BASE/miniforge3/bin/python3
RESULT_ROOT=$BASE/artifacts/r12/ssqac_proof_carrying

: "${PREPARATION:?PREPARATION must point to a sealed preparation artifact}"
SEED=${SEED:-20260724}
UPDATES=${UPDATES:-1500}
BATCH=${BATCH:-4}
LR=${LR:-0.0006}
FIELD_WIDTH=${FIELD_WIDTH:-64}
WIDTH=${WIDTH:-384}
CELL_HIDDEN=${CELL_HIDDEN:-512}
MATRIX_LAYERS=${MATRIX_LAYERS:-4}
CONTRACT_HIDDEN=${CONTRACT_HIDDEN:-512}
COORDINATE_HARMONICS=${COORDINATE_HARMONICS:-4}
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

echo "[ssqac-proof] node=${SLURMD_NODENAME:-?} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[ssqac-proof] preparation_sha256=$(sha256sum "$PREPARATION" | awk '{print $1}')"
nvidia-smi -L
"$PYTHON" "$BASE/train/ssqac_proof_carrying_controller.py" run \
  --preparation "$PREPARATION" \
  --seed "$SEED" \
  --optimizer-updates "$UPDATES" \
  --batch-size "$BATCH" \
  --learning-rate "$LR" \
  --maximum-rollout-steps "$MAX_STEPS" \
  --field-width "$FIELD_WIDTH" \
  --width "$WIDTH" \
  --cell-hidden "$CELL_HIDDEN" \
  --matrix-layers "$MATRIX_LAYERS" \
  --contract-hidden "$CONTRACT_HIDDEN" \
  --coordinate-harmonics "$COORDINATE_HARMONICS" \
  --device cuda \
  --output "$OUTPUT" \
  --model-dir "$MODEL_DIR"

test -s "$OUTPUT"
sha256sum "$OUTPUT"
echo "[ssqac-proof] report=$OUTPUT"
