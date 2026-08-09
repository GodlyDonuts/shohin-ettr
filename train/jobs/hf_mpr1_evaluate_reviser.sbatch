#!/bin/bash
#SBATCH --job-name=mpr1-eval
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/mpr1_eval_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/mpr1_eval_%j.err

set -euo pipefail
: "${RUNTIME:?RUNTIME is required}"
: "${ADAPTER_CHECKPOINT:?ADAPTER_CHECKPOINT is required}"
test -f "$ADAPTER_CHECKPOINT"
export ADAPTER_CHECKPOINT_SHA256
ADAPTER_CHECKPOINT_SHA256=$(sha256sum "$ADAPTER_CHECKPOINT" | cut -d' ' -f1)
exec bash "$RUNTIME/train/jobs/hf_idr1_evaluate_reviser.sbatch"

