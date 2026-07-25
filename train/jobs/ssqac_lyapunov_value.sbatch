#!/bin/bash
#SBATCH -J ssqac-lyapunov
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 16:00:00
#SBATCH --exclude=evc31,evc33,evc36,evc43,evc44
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/ssqac_lyapunov_%j.out
#SBATCH --open-mode=append
#
# GPU fit/evaluation only. CPU preparation must already have produced PACKET.
# The candidate never loads Shohin or any checkpoint and writes only its own
# isolated report/model directory.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=$BASE/miniforge3/bin/python3
RESULT_ROOT=$BASE/artifacts/r12/ssqac_lyapunov_value
OUTPUT=$RESULT_ROOT/report_${SLURM_JOB_ID}.json
MODEL_DIR=$RESULT_ROOT/models_${SLURM_JOB_ID}

: "${PACKET:?PACKET must name one CPU-prepared packet}"
test -s "$PACKET"
test ! -e "$OUTPUT"
test ! -e "$MODEL_DIR"
mkdir -p "$BASE/logs" "$RESULT_ROOT"

UPDATES=${UPDATES:-1600}
NODE_BATCH=${NODE_BATCH:-256}
DECISION_BATCH=${DECISION_BATCH:-24}
LR=${LR:-0.0004}
FIELD_WIDTH=${FIELD_WIDTH:-64}
WIDTH=${WIDTH:-512}
CELL_HIDDEN=${CELL_HIDDEN:-1024}
MATRIX_LAYERS=${MATRIX_LAYERS:-6}
STATE_HIDDEN=${STATE_HIDDEN:-768}
COORDINATE_HARMONICS=${COORDINATE_HARMONICS:-6}

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=$BASE:$BASE/train
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[ssqac-lyapunov] node=${SLURMD_NODENAME:-?} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[ssqac-lyapunov] packet_sha256=$(sha256sum "$PACKET" | cut -d' ' -f1)"
nvidia-smi -L
"$PYTHON" "$BASE/train/ssqac_lyapunov_value_controller.py" \
  --prepared-packet "$PACKET" \
  --optimizer-updates "$UPDATES" \
  --node-batch-size "$NODE_BATCH" \
  --decision-batch-size "$DECISION_BATCH" \
  --learning-rate "$LR" \
  --field-width "$FIELD_WIDTH" \
  --width "$WIDTH" \
  --cell-hidden "$CELL_HIDDEN" \
  --matrix-layers "$MATRIX_LAYERS" \
  --state-hidden "$STATE_HIDDEN" \
  --coordinate-harmonics "$COORDINATE_HARMONICS" \
  --device cuda \
  --model-dir "$MODEL_DIR" \
  --output "$OUTPUT"

test ! -e "$PACKET"
test -s "$OUTPUT"
echo "[ssqac-lyapunov] report=$OUTPUT"
