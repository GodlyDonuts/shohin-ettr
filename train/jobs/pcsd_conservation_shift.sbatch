#!/bin/bash
#SBATCH --job-name=pcsd-shift
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --exclude=evc33,evc43
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/pcsd_shift_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/pcsd_shift_%j.err

# One isolated arm of the bounded Conservation-Shift falsifier. RUNTIME must
# be a read-only, commit-addressed source snapshot; OUT_ROOT must be unique.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=$BASE/miniforge3/bin/python3
: "${RUNTIME:?set immutable runtime root}"
: "${OUT_ROOT:?set unique result root}"
: "${ARM:?set pcsd or dense}"

SEED=${SEED:-31}
DATA_SEED=${DATA_SEED:-20260806}
UPDATES=${UPDATES:-4000}
BATCH_SIZE=${BATCH_SIZE:-256}
EVAL_COUNT=${EVAL_COUNT:-4096}
TRAIN_DEPTH_MAX=${TRAIN_DEPTH_MAX:-8}
LOG_EVERY=${LOG_EVERY:-100}
REPORT=$OUT_ROOT/${ARM}_seed${SEED}.json
MANIFEST=$RUNTIME/artifacts/diagnostics/pcsd_conservation_shift_cohort_v1.json

test -f "$RUNTIME/train/pcsd_conservation_shift.py"
test -f "$RUNTIME/train/prompt_conditioned_syndrome.py"
test -f "$MANIFEST"
test ! -e "$REPORT"
test ! -e "${REPORT%.json}.pt"
mkdir -p "$BASE/logs" "$OUT_ROOT"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=$RUNTIME/train
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[pcsd-shift] job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-?} arm=$ARM seed=$SEED"
nvidia-smi -L
"$PYTHON" "$RUNTIME/train/pcsd_conservation_shift.py" \
  --arm "$ARM" \
  --output "$REPORT" \
  --seed "$SEED" \
  --data-seed "$DATA_SEED" \
  --updates "$UPDATES" \
  --batch-size "$BATCH_SIZE" \
  --train-depth-max "$TRAIN_DEPTH_MAX" \
  --eval-count "$EVAL_COUNT" \
  --log-every "$LOG_EVERY" \
  --cohort-manifest "$MANIFEST" \
  --evaluation-split development

test -s "$REPORT"
test -s "${REPORT%.json}.pt"
echo "[pcsd-shift] report=$REPORT checkpoint=${REPORT%.json}.pt"
