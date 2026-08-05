#!/bin/bash
#SBATCH --job-name=fcpt-plural
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --exclude=evc33,evc43
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/fcpt_plural_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/fcpt_plural_%j.err

set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=$BASE/miniforge3/bin/python3
: "${RUNTIME:?set immutable runtime root}"
: "${OUT_ROOT:?set unique result root}"
: "${ARM:?set fcpt, selection, independent, or soft}"

SEED=${SEED:-17}
DATA_SEED=${DATA_SEED:-20260850}
UPDATES=${UPDATES:-1000}
BATCH_SIZE=${BATCH_SIZE:-256}
EVAL_COUNT=${EVAL_COUNT:-1024}
LOG_EVERY=${LOG_EVERY:-50}
BEHAVIOR_OBJECTIVE=${BEHAVIOR_OBJECTIVE:-coverage}
DIVERSITY_WEIGHT=${DIVERSITY_WEIGHT:-0.2}
REPORT=$OUT_ROOT/${ARM}_seed${SEED}.json
MANIFEST=$RUNTIME/artifacts/diagnostics/fcpt_plural_cohort_v1.json

test -f "$RUNTIME/train/falsification_coupled_particles.py"
test -f "$RUNTIME/train/fcpt_plural_reasoning.py"
test -f "$MANIFEST"
test ! -e "$REPORT"
test ! -e "${REPORT%.json}.pt"
mkdir -p "$BASE/logs" "$OUT_ROOT"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=$RUNTIME/train
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[fcpt-plural] job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-?} arm=$ARM seed=$SEED"
nvidia-smi -L
"$PYTHON" "$RUNTIME/train/fcpt_plural_reasoning.py" \
  --arm "$ARM" \
  --output "$REPORT" \
  --seed "$SEED" \
  --data-seed "$DATA_SEED" \
  --updates "$UPDATES" \
  --batch-size "$BATCH_SIZE" \
  --eval-count "$EVAL_COUNT" \
  --log-every "$LOG_EVERY" \
  --behavior-objective "$BEHAVIOR_OBJECTIVE" \
  --diversity-weight "$DIVERSITY_WEIGHT" \
  --cohort-manifest "$MANIFEST"

test -s "$REPORT"
test -s "${REPORT%.json}.pt"
echo "[fcpt-plural] report=$REPORT checkpoint=${REPORT%.json}.pt"
