#!/bin/bash
#SBATCH --job-name=mfm-shohin-smoke
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=scratchpad/slurm-%j.out
#SBATCH --exclude=evc26,evc31,evc33,evc44

set -euo pipefail

ROOT="${ROOT:-$PWD}"
OUTPUT="${OUTPUT:-artifacts/r12/source_deleted_multifamily_machine_board_v1/connected_smoke_seed20260725.json}"

cd "$ROOT"
test -f train/flagship_out/ckpt_0300000.pt
test -f artifacts/tokenizer/tokenizer.json
mkdir -p "$(dirname "$OUTPUT")" scratchpad

srun python train/run_multifamily_shohin_connected_smoke.py \
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
