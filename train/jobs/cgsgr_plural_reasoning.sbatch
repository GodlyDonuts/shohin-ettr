#!/bin/bash
#SBATCH --job-name=cgsgr-plural
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --exclude=evc33,evc43
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/cgsgr_plural_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/cgsgr_plural_%j.err

set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=$BASE/miniforge3/bin/python3
: "${RUNTIME:?set immutable runtime root}"
: "${OUT_ROOT:?set unique result root}"
: "${ARM:?set guided, fixed, or dense}"

SEED=${SEED:-23}
DATA_SEED=${DATA_SEED:-20260870}
UPDATES=${UPDATES:-1000}
BATCH_SIZE=${BATCH_SIZE:-256}
EVAL_COUNT=${EVAL_COUNT:-1024}
LOG_EVERY=${LOG_EVERY:-50}
REPORT=$OUT_ROOT/${ARM}_seed${SEED}.json
MANIFEST=$RUNTIME/artifacts/diagnostics/fcpt_plural_cohort_v1.json

test -f "$RUNTIME/train/counterexample_guided_revision.py"
test -f "$RUNTIME/train/cgsgr_plural_reasoning.py"
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

echo "[cgsgr-plural] job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-?} arm=$ARM seed=$SEED"
nvidia-smi -L
"$PYTHON" "$RUNTIME/train/cgsgr_plural_reasoning.py" \
  --arm "$ARM" \
  --output "$REPORT" \
  --seed "$SEED" \
  --data-seed "$DATA_SEED" \
  --updates "$UPDATES" \
  --batch-size "$BATCH_SIZE" \
  --eval-count "$EVAL_COUNT" \
  --log-every "$LOG_EVERY" \
  --cohort-manifest "$MANIFEST"

test -s "$REPORT"
test -s "${REPORT%.json}.pt"
echo "[cgsgr-plural] report=$REPORT checkpoint=${REPORT%.json}.pt"
