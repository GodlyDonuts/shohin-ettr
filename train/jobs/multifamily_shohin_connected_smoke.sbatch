#!/bin/bash
#SBATCH -J mfm-shohin-smoke
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/multifamily_shohin_smoke_%j.out
#SBATCH --open-mode=append
#SBATCH --exclude=evc26,evc31,evc33,evc44

set -euo pipefail

ROOT="${ROOT:-/lustre/fs1/home/sa305415/shohin}"
PYTHON="$ROOT/miniforge3/bin/python3"
OUTPUT="${OUTPUT:-artifacts/r12/source_deleted_multifamily_machine_board_v1/connected_smoke_seed20260725.json}"

cd "$ROOT"
test -f train/flagship_out/ckpt_0300000.pt
test -f artifacts/tokenizer/tokenizer.json
mkdir -p "$(dirname "$OUTPUT")" scratchpad
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT:$ROOT/train"
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[multifamily-shohin-smoke] node=${SLURMD_NODENAME:-?} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
srun "$PYTHON" train/run_multifamily_shohin_connected_smoke.py \
  --checkpoint train/flagship_out/ckpt_0300000.pt \
  --tokenizer artifacts/tokenizer/tokenizer.json \
  --block-indices 17,25,29 \
  --seed 20260725 \
  --steps 300 \
  --width 96 \
  --layers 2 \
  --learning-rate 0.003 \
  --feature-chunk-size 64 \
  --device cuda \
  --output "$OUTPUT"
