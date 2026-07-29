#!/bin/bash
# Parallelize one exact frozen main materialization cell into an isolated root.
# The original selected shard and frozen materializer remain the only semantics.
#SBATCH -J ettr-v3-mat-fast
#SBATCH -p normal
#SBATCH --account=skattel
#SBATCH --qos=normal
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 16:00:00
#SBATCH -o /lustre/fs1/home/sa305415/ettr-il-v3/logs/materialize-fast-%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue

set -euo pipefail
umask 077
export PATH=/usr/bin:/bin
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
PY=${PY:-$BASE/miniforge3/bin/python}
ADAPTER_ROOT=${ADAPTER_ROOT:?set the exact immutable adapter runtime}
SOURCE_ROOT=${SOURCE_ROOT:?set the exact frozen materializer source root}
TASK_MANIFEST=${TASK_MANIFEST:?set the immutable main task manifest}
SELECTED_ROOT=${SELECTED_ROOT:?set the immutable selected main root}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a fresh isolated output root}
REPORTS_ROOT=${REPORTS_ROOT:?set a fresh isolated reports root}
TOKENIZER=${TOKENIZER:?set the exact immutable tokenizer}
MATERIALIZER_FREEZE=${MATERIALIZER_FREEZE:?set the exact materializer freeze}
TASK_INDEX=${TASK_INDEX:?set one main task index}
EXECUTION_SOURCE_COMMIT=${EXECUTION_SOURCE_COMMIT:?set the adapter source commit}
ADAPTER_CODE=$ADAPTER_ROOT/accelerate_ettr_il_v3_materialization.py
ADAPTER_SHA256=${ADAPTER_SHA256:?set the exact adapter code SHA-256}
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-16}}

for path in \
  "$ADAPTER_ROOT" \
  "$SOURCE_ROOT" \
  "$TASK_MANIFEST" \
  "$SELECTED_ROOT" \
  "$OUTPUT_ROOT" \
  "$REPORTS_ROOT" \
  "$TOKENIZER" \
  "$MATERIALIZER_FREEZE" \
  "$ADAPTER_CODE" \
  "$PY"; do
  [[ "$path" == /* ]] || {
    echo "[ettr-fast] all paths must be absolute" >&2
    exit 2
  }
done
if [[ ! -d "$ADAPTER_ROOT" || -L "$ADAPTER_ROOT" \
  || ! -s "$ADAPTER_ROOT/SHA256SUMS" \
  || ! -d "$SOURCE_ROOT" || -L "$SOURCE_ROOT" \
  || ! -f "$TASK_MANIFEST" || -L "$TASK_MANIFEST" \
  || ! -d "$SELECTED_ROOT" || -L "$SELECTED_ROOT" \
  || -e "$OUTPUT_ROOT" || -L "$OUTPUT_ROOT" \
  || -e "$REPORTS_ROOT" || -L "$REPORTS_ROOT" \
  || ! -f "$TOKENIZER" || -L "$TOKENIZER" \
  || ! -f "$MATERIALIZER_FREEZE" || -L "$MATERIALIZER_FREEZE" \
  || ! -f "$ADAPTER_CODE" || -L "$ADAPTER_CODE" \
  || ! -x "$PY" ]]; then
  echo "[ettr-fast] preflight differs" >&2
  exit 2
fi
if [[ ! "$TASK_INDEX" =~ ^[0-9]+$ \
  || ! "$WORKERS" =~ ^[0-9]+$ \
  || ! "$EXECUTION_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ \
  || ! "$ADAPTER_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "[ettr-fast] scalar contract differs" >&2
  exit 2
fi
(cd "$ADAPTER_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
printf '%s  %s\n' "$ADAPTER_SHA256" "$ADAPTER_CODE" | sha256sum -c -
mkdir -m 700 "$OUTPUT_ROOT" "$REPORTS_ROOT"
export PYTHONPATH="$SOURCE_ROOT/pipeline"
"$PY" -u "$ADAPTER_CODE" \
  --task-manifest "$TASK_MANIFEST" \
  --selected-root "$SELECTED_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --reports-root "$REPORTS_ROOT" \
  --tokenizer "$TOKENIZER" \
  --materializer-source-root "$SOURCE_ROOT" \
  --materializer-freeze "$MATERIALIZER_FREEZE" \
  --task-index "$TASK_INDEX" \
  --workers "$WORKERS" \
  --execution-source-commit "$EXECUTION_SOURCE_COMMIT" \
  --execution-code-sha256 "$ADAPTER_SHA256"
