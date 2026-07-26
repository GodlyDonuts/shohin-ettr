#!/bin/bash
# Isolated ETTR resource profile. This wrapper never submits another job,
# invokes a trainer, reads training shards, or writes a model checkpoint.
#SBATCH --job-name=ettr-h100-profile
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
#SBATCH --output=logs/ettr_h100_profile_%j.out
#SBATCH --error=logs/ettr_h100_profile_%j.err

set -euo pipefail

CODE_ROOT=${CODE_ROOT:?set CODE_ROOT to the clean reviewed source checkout}
SOURCE_COMMIT=${SOURCE_COMMIT:?set SOURCE_COMMIT to the exact reviewed commit}
CHECKPOINT=${CHECKPOINT:?set CHECKPOINT to the read-only base checkpoint}
CHECKPOINT_SHA256=${CHECKPOINT_SHA256:?set the exact checkpoint SHA-256}
CHECKPOINT_STEP=${CHECKPOINT_STEP:?set the exact checkpoint step}
OUTDIR=${OUTDIR:?set OUTDIR to a fresh isolated profile directory}
PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python}

if [[ -n "${SHARDS:-}" ]]; then
  echo "live SHARDS are forbidden in ETTR profiling" >&2
  exit 2
fi
unset SHARDS

for path in "$CODE_ROOT" "$CHECKPOINT" "$OUTDIR" "$PYTHON"; do
  if [[ "$path" != /* ]]; then
    echo "ETTR profile paths must be absolute: $path" >&2
    exit 2
  fi
done
if [[ ! "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ETTR profile source commit differs" >&2
  exit 2
fi
if [[ ! "$CHECKPOINT_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "ETTR profile checkpoint hash differs" >&2
  exit 2
fi
if [[ ! "$CHECKPOINT_STEP" =~ ^[0-9]+$ ]]; then
  echo "ETTR profile checkpoint step differs" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT" || -L "$CHECKPOINT" ]]; then
  echo "ETTR profile checkpoint must be a physical regular file" >&2
  exit 2
fi
if [[ -e "$OUTDIR" || -L "$OUTDIR" ]]; then
  echo "refusing existing ETTR profile output: $OUTDIR" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "ETTR profile Python is unavailable" >&2
  exit 2
fi

cd "$CODE_ROOT"
if [[ "$(git rev-parse HEAD)" != "$SOURCE_COMMIT" ]]; then
  echo "ETTR profile source checkout differs" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "ETTR profile source checkout is not clean" >&2
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

"$PYTHON" -u train/profile_ettr_h100.py \
  --mode h100 \
  --output-dir "$OUTDIR" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --expected-step "$CHECKPOINT_STEP" \
  --batch-size "${BATCH_SIZE:-1}" \
  --microsteps "${MICROSTEPS:-2}" \
  --warmup-updates "${WARMUP_UPDATES:-2}" \
  --measured-updates "${MEASURED_UPDATES:-10}" \
  --world-tokens "${WORLD_TOKENS:-256}" \
  --command-tokens "${COMMAND_TOKENS:-64}" \
  --query-tokens "${QUERY_TOKENS:-128}" \
  --reactor-steps "${REACTOR_STEPS:-4}" \
  --train-scope "${TRAIN_SCOPE:-architecture}" \
  --compile-mode "${COMPILE_MODE:-default}"

test -r "$OUTDIR/report.json"
sha256sum "$OUTDIR/report.json"
