#!/bin/bash
# Isolated production ETTR step canary. No training shards or checkpoints are
# written, and the protected base is opened read-only.
#SBATCH --job-name=ettr-prod-canary
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=truncate
#SBATCH --output=logs/ettr_prod_canary_%j.out
#SBATCH --error=logs/ettr_prod_canary_%j.err

set -euo pipefail

CODE_ROOT=${CODE_ROOT:?set CODE_ROOT}
SOURCE_COMMIT=${SOURCE_COMMIT:?set SOURCE_COMMIT}
CHECKPOINT=${CHECKPOINT:?set CHECKPOINT}
CHECKPOINT_SHA256=${CHECKPOINT_SHA256:?set CHECKPOINT_SHA256}
OUTDIR=${OUTDIR:?set OUTDIR}
PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python}

if [[ -n "${SHARDS:-}" ]]; then
  echo "SHARDS are forbidden in the ETTR production canary" >&2
  exit 2
fi
unset SHARDS

for path in "$CODE_ROOT" "$CHECKPOINT" "$OUTDIR" "$PYTHON"; do
  if [[ "$path" != /* ]]; then
    echo "canary paths must be absolute: $path" >&2
    exit 2
  fi
done
if [[ ! "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "source commit differs" >&2
  exit 2
fi
if [[ ! "$CHECKPOINT_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "checkpoint SHA-256 differs" >&2
  exit 2
fi
if [[ ! -r "$CHECKPOINT" || -L "$CHECKPOINT" ]]; then
  echo "checkpoint must be a physical readable file" >&2
  exit 2
fi
if [[ -e "$OUTDIR" || -L "$OUTDIR" ]]; then
  echo "refusing existing canary output: $OUTDIR" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "canary Python is unavailable" >&2
  exit 2
fi

export PYTHONPATH="$CODE_ROOT/train"
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON" -u "$CODE_ROOT/train/canary_ettr_production_step_h100.py" \
  --output "$OUTDIR" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --expected-step 300000 \
  --source-commit "$SOURCE_COMMIT" \
  --compile-mode "${COMPILE_MODE:-default}"

test -r "$OUTDIR/report.json"
sha256sum "$OUTDIR/report.json"
