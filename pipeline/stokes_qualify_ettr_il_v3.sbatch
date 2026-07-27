#!/usr/bin/env bash
#SBATCH --job-name=ettr-v3-qual
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
: "${ETTR_V3_CANDIDATE_ROOT:?ETTR_V3_CANDIDATE_ROOT is required}"
: "${ETTR_V3_QUALIFIED_ROOT:?ETTR_V3_QUALIFIED_ROOT is required}"
: "${ETTR_V3_TOKENIZER:?ETTR_V3_TOKENIZER is required}"
: "${ETTR_V3_SOURCE_COMMIT:?ETTR_V3_SOURCE_COMMIT is required}"
: "${ETTR_V3_QUALIFIER_FREEZE:?ETTR_V3_QUALIFIER_FREEZE is required}"
: "${ETTR_V3_QUALIFICATION_ROLE:?ETTR_V3_QUALIFICATION_ROLE is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

PYTHON_BIN="${ETTR_V3_PYTHON_BIN:?ETTR_V3_PYTHON_BIN is required}"
test -x "${PYTHON_BIN}"
test -f "${ETTR_V3_TOKENIZER}"
mkdir -p \
  "${ETTR_V3_QUALIFIED_ROOT}/shards" \
  "${ETTR_V3_QUALIFIED_ROOT}/reports"

arguments=(
  --candidates "${ETTR_V3_CANDIDATE_ROOT}"
  --output-root "${ETTR_V3_QUALIFIED_ROOT}"
  --tokenizer "${ETTR_V3_TOKENIZER}"
  --qualifier-source-root "${ETTR_V3_SOURCE_ROOT}"
  --qualifier-source-commit "${ETTR_V3_SOURCE_COMMIT}"
  --qualifier-freeze "${ETTR_V3_QUALIFIER_FREEZE}"
  --matrix-index "${SLURM_ARRAY_TASK_ID}"
  --role "${ETTR_V3_QUALIFICATION_ROLE}"
)
if [[ -n "${ETTR_V3_CONFIRMATION_KEY_FILE:-}" ]]; then
  arguments+=(--confirmation-key-file "${ETTR_V3_CONFIRMATION_KEY_FILE}")
fi

export PYTHONPATH="${ETTR_V3_SOURCE_ROOT}/pipeline"
exec "${PYTHON_BIN}" \
  "${ETTR_V3_SOURCE_ROOT}/pipeline/qualify_ettr_il_v3_candidates.py" \
  "${arguments[@]}"
