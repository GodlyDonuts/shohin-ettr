#!/bin/bash
#SBATCH --job-name=csdc-semantic
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --exclude=evc33,evc43
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/csdc_semantic_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/csdc_semantic_%j.err

set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=$BASE/miniforge3/bin/python3
: "${RUNTIME:?set immutable runtime root}"
: "${OUT_ROOT:?set unique result root}"
: "${REASONER_CHECKPOINT:?set frozen learned-PSPA checkpoint}"

SEED=${SEED:-53}
UPDATES=${UPDATES:-1500}
BATCH_SIZE=${BATCH_SIZE:-128}
EVAL_COUNT=${EVAL_COUNT:-1024}
LOG_EVERY=${LOG_EVERY:-50}
REPORT=$OUT_ROOT/seed${SEED}.json

for path in \
  train/prompt_selected_presented_algebra.py \
  train/pspa_presented_reasoning.py \
  train/learned_pspa_language_reasoning.py \
  train/evaluate_deferred_pspa_closure.py \
  train/evaluate_counterexample_selected_closure.py \
  train/csdc_semantic_bridge.py; do
  test -f "$RUNTIME/$path"
done
test -s "$REASONER_CHECKPOINT"
test ! -e "$REPORT"
test ! -e "${REPORT%.json}.pt"
mkdir -p "$BASE/logs" "$OUT_ROOT"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=$RUNTIME/train
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[csdc-semantic] job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-?} seed=$SEED"
nvidia-smi -L
"$PYTHON" "$RUNTIME/train/csdc_semantic_bridge.py" \
  --reasoner-checkpoint "$REASONER_CHECKPOINT" \
  --output "$REPORT" \
  --seed "$SEED" \
  --updates "$UPDATES" \
  --batch-size "$BATCH_SIZE" \
  --eval-count "$EVAL_COUNT" \
  --log-every "$LOG_EVERY"

test -s "$REPORT"
test -s "${REPORT%.json}.pt"
echo "[csdc-semantic] report=$REPORT checkpoint=${REPORT%.json}.pt"

