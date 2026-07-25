#!/usr/bin/env bash
#SBATCH --job-name=ssqac-search-scale
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/ssqac_search_scale_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/ssqac_search_scale_%j.err
#SBATCH --exclude=evc26,evc31,evc33,evc44

set -euo pipefail

ROOT=/lustre/fs1/home/sa305415/shohin
SEED=${SEED:?SEED is required}
OUT_ROOT="$ROOT/artifacts/r12/ssqac_search_distillation_scale_v3_consensus"
OUTPUT="$OUT_ROOT/report_seed${SEED}_${SLURM_JOB_ID}.json"
SCRATCH="${SLURM_TMPDIR:-/tmp}/ssqac_search_scale_${SLURM_JOB_ID}"
SOURCE_SHA256=8664339392d7b23de114a4316bb0f37e5ba25b3827aae330cc10a82ce535da4b

mkdir -p "$ROOT/logs" "$OUT_ROOT" "$SCRATCH"
test ! -e "$OUTPUT"

cd "$ROOT"
test "$(sha256sum pipeline/ssqac_search_distillation_pilot.py | cut -d' ' -f1)" = \
  "$SOURCE_SHA256"
export PYTHONPATH="$ROOT:$ROOT/train"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

printf '[ssqac-search-scale] node=%s seed=%s gpu=%s\n' \
  "$(hostname)" "$SEED" "${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,uuid --format=csv,noheader

"$ROOT/miniforge3/bin/python3" \
  "$ROOT/pipeline/ssqac_search_distillation_pilot.py" \
  --seed "$SEED" \
  --device cuda \
  --train-cases 1024 \
  --evaluation-cases 512 \
  --states-per-case 24 \
  --fit-maximum-rows 4 \
  --fit-maximum-columns 6 \
  --evaluation-minimum-rows 5 \
  --evaluation-minimum-columns 7 \
  --evaluation-maximum-rows 5 \
  --evaluation-maximum-columns 8 \
  --maximum-rows 6 \
  --maximum-columns 8 \
  --width 384 \
  --blocks 6 \
  --feedforward 1536 \
  --epochs 60 \
  --batch-size 512 \
  --learning-rate 0.0003 \
  --weight-decay 0.01 \
  --amp-bfloat16 \
  --torch-compile \
  --search-max-nodes 8192 \
  --search-max-edges 200000 \
  --search-max-depth 48 \
  --search-max-frontier 128 \
  --search-beam-width 128 \
  --search-max-instructions 256 \
  --search-policy-noise 8.0 \
  --maximum-rollout-macros 192 \
  --maximum-rollout-instructions 640 \
  --minimum-material-evaluation-cases 512 \
  --minimum-material-certification-rate 0.20 \
  --search-beats-oracle-margin 0.05 \
  --search-beats-random-margin 0.10 \
  --scratch-root "$SCRATCH" \
  --output "$OUTPUT"

sha256sum "$OUTPUT"
