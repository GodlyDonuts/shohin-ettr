#!/bin/bash
#SBATCH -J episodic-generator-constraint
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/episodic_generator_constraint_%j.out
#SBATCH --open-mode=append
#SBATCH --exclude=evc26,evc31,evc33,evc44

set -euo pipefail

ROOT="${ROOT:-/lustre/fs1/home/sa305415/shohin}"
PYTHON="$ROOT/miniforge3/bin/python3"
SEED="${SEED:-20260725}"
OUTPUT="$ROOT/artifacts/r12/episodic_generator_constraint/seed${SEED}.json"
MODEL="$ROOT/artifacts/r12/episodic_generator_constraint/seed${SEED}.pt"

cd "$ROOT"
mkdir -p "$(dirname "$OUTPUT")" logs
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT:$ROOT/train:$ROOT/pipeline"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

"$PYTHON" train/run_episodic_generator_constraint.py \
  --seed "$SEED" \
  --steps 1000 \
  --width 128 \
  --layers 2 \
  --heads 4 \
  --learning-rate 0.002 \
  --batch-size 64 \
  --auxiliary-rows 300 \
  --device cuda \
  --model-output "$MODEL" \
  --output "$OUTPUT"
