#!/bin/bash
#SBATCH --job-name=temporal-eval
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/temporal_eval_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/temporal_eval_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${EVALUATOR:?EVALUATOR is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"
test -f "$CHECKPOINT"
export ADAPTER_CHECKPOINT=$CHECKPOINT
export ADAPTER_CHECKPOINT_SHA256
ADAPTER_CHECKPOINT_SHA256=$(sha256sum "$CHECKPOINT" | cut -d' ' -f1)

case "$EVALUATOR" in
  sctr1)
    exec bash "$RUNTIME/train/jobs/hf_sctr1_evaluate.sbatch"
    ;;
  idr1)
    exec bash "$RUNTIME/train/jobs/hf_idr1_evaluate_reviser.sbatch"
    ;;
  *)
    echo "unknown temporal evaluator: $EVALUATOR" >&2
    exit 4
    ;;
esac
