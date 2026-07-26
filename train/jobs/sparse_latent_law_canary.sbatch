#!/bin/bash
#SBATCH -J sparse-law-canary
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/sparse_law_canary_%j.out
#SBATCH --open-mode=append
#SBATCH --exclude=evc26,evc31,evc33,evc44

set -euo pipefail

ROOT="${ROOT:-/lustre/fs1/home/sa305415/shohin}"
PYTHON="$ROOT/miniforge3/bin/python3"
SEED="${SEED:-20260725}"
STEPS="${STEPS:-2000}"
WIDTH="${WIDTH:-128}"
LAYERS="${LAYERS:-2}"
HEADS="${HEADS:-4}"
BATCH="${BATCH:-64}"
AUXILIARY="${AUXILIARY:-3000}"
ARCHITECTURE="${ARCHITECTURE:-microcoded}"
OUTPUT="$ROOT/artifacts/r12/sparse_latent_law_curriculum/seed${SEED}_${ARCHITECTURE}.json"
MODEL="$ROOT/artifacts/r12/sparse_latent_law_curriculum/seed${SEED}_${ARCHITECTURE}.pt"

cd "$ROOT"
mkdir -p "$(dirname "$OUTPUT")" logs
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT:$ROOT/train:$ROOT/pipeline"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

"$PYTHON" train/run_sparse_latent_law_curriculum.py \
  --seed "$SEED" \
  --steps "$STEPS" \
  --width "$WIDTH" \
  --layers "$LAYERS" \
  --heads "$HEADS" \
  --learning-rate 0.002 \
  --batch-size "$BATCH" \
  --auxiliary-rows "$AUXILIARY" \
  --architecture "$ARCHITECTURE" \
  --device cuda \
  --model-output "$MODEL" \
  --output "$OUTPUT"
