#!/bin/bash
#SBATCH --job-name=nemotron-scale-basis
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/nemotron_scale_basis_%j.out
#SBATCH --error=logs/nemotron_scale_basis_%j.err

set -euo pipefail

: "${PYTHON:?PYTHON is required}"
: "${RUNTIME:?RUNTIME is required}"
: "${SUPER_ROOT:?SUPER_ROOT is required}"
: "${ULTRA_BASIS_ROOT:?ULTRA_BASIS_ROOT is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

[[ -x "$PYTHON" ]]
[[ -d "$RUNTIME" && ! -L "$RUNTIME" ]]
[[ -d "$SUPER_ROOT" && ! -L "$SUPER_ROOT" ]]
[[ -d "$ULTRA_BASIS_ROOT" && ! -L "$ULTRA_BASIS_ROOT" ]]
[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]]

mkdir -p logs "$OUTPUT_ROOT"
trap 'chmod -R a-w "$OUTPUT_ROOT" 2>/dev/null || true' EXIT
export PATH=/apps/slurm/current/bin:/usr/bin:/bin
export PYTHONPATH="$RUNTIME/pipeline"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=16
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false

"$PYTHON" -P -s -B "$RUNTIME/pipeline/build_nemotron_scale_transfer_basis.py" \
  --super-embedding "$SUPER_ROOT/model-00001-of-00026.safetensors" \
  --ultra-embedding "$ULTRA_BASIS_ROOT/model-00001-of-00113.safetensors" \
  --super-tokenizer "$SUPER_ROOT/tokenizer.json" \
  --ultra-tokenizer "$ULTRA_BASIS_ROOT/tokenizer.json" \
  --exclude-token-id 0 \
  --exclude-token-id 128001 \
  --exclude-token-id 128008 \
  --output-factor "$OUTPUT_ROOT/factor.pt" \
  --output-report "$OUTPUT_ROOT/report.json"

[[ -f "$OUTPUT_ROOT/factor.pt" && ! -L "$OUTPUT_ROOT/factor.pt" ]]
[[ -f "$OUTPUT_ROOT/report.json" && ! -L "$OUTPUT_ROOT/report.json" ]]
