#!/usr/bin/env bash
#SBATCH --job-name=ettr-v3-mat
#SBATCH --partition=normal
#SBATCH --account=skattel
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --no-requeue

set -euo pipefail
umask 077

: "${ETTR_V3_SOURCE_ROOT:?ETTR_V3_SOURCE_ROOT is required}"
: "${ETTR_V3_TASK_MANIFEST:?ETTR_V3_TASK_MANIFEST is required}"
: "${ETTR_V3_SELECTED_ROOT:?ETTR_V3_SELECTED_ROOT is required}"
: "${ETTR_V3_OUTPUT_ROOT:?ETTR_V3_OUTPUT_ROOT is required}"
: "${ETTR_V3_REPORTS_ROOT:?ETTR_V3_REPORTS_ROOT is required}"
: "${ETTR_V3_TOKENIZER:?ETTR_V3_TOKENIZER is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

PYTHON_BIN="${ETTR_V3_PYTHON_BIN:?ETTR_V3_PYTHON_BIN is required}"
test -x "${PYTHON_BIN}"
test -f "${ETTR_V3_TASK_MANIFEST}"
test -f "${ETTR_V3_TOKENIZER}"

arguments=(
  worker
  --task-manifest "${ETTR_V3_TASK_MANIFEST}"
  --selected-root "${ETTR_V3_SELECTED_ROOT}"
  --output-root "${ETTR_V3_OUTPUT_ROOT}"
  --reports-root "${ETTR_V3_REPORTS_ROOT}"
  --tokenizer "${ETTR_V3_TOKENIZER}"
  --task-index "${SLURM_ARRAY_TASK_ID}"
)
if [[ -n "${ETTR_V3_CONFIRMATION_KEY_FILE:-}" ]]; then
  arguments+=(--confirmation-key-file "${ETTR_V3_CONFIRMATION_KEY_FILE}")
fi

export PYTHONPATH="${ETTR_V3_SOURCE_ROOT}/pipeline"
exec "${PYTHON_BIN}" \
  "${ETTR_V3_SOURCE_ROOT}/pipeline/materialize_ettr_il_v3_corpus.py" \
  "${arguments[@]}"
