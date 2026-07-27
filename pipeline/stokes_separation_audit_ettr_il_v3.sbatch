#!/usr/bin/env bash
#SBATCH --job-name=ettr-v3-separate
#SBATCH --partition=normal
#SBATCH --account=skattel
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --no-requeue

set -euo pipefail
umask 077

: "${ETTR_V3_SOURCE_ROOT:?ETTR_V3_SOURCE_ROOT is required}"
: "${ETTR_V3_MAIN_AUDIT:?ETTR_V3_MAIN_AUDIT is required}"
: "${ETTR_V3_CONFIRMATION_AUDIT:?ETTR_V3_CONFIRMATION_AUDIT is required}"
: "${ETTR_V3_MAIN_OUTPUT_ROOT:?ETTR_V3_MAIN_OUTPUT_ROOT is required}"
: "${ETTR_V3_CONFIRMATION_OUTPUT_ROOT:?ETTR_V3_CONFIRMATION_OUTPUT_ROOT is required}"
: "${ETTR_V3_SEPARATION_REPORT:?ETTR_V3_SEPARATION_REPORT is required}"
: "${ETTR_V3_PYTHON_BIN:?ETTR_V3_PYTHON_BIN is required}"

export PYTHONPATH="${ETTR_V3_SOURCE_ROOT}/pipeline"
exec "${ETTR_V3_PYTHON_BIN}" \
  "${ETTR_V3_SOURCE_ROOT}/pipeline/materialize_ettr_il_v3_corpus.py" \
  separation-audit \
  --main-audit "${ETTR_V3_MAIN_AUDIT}" \
  --confirmation-audit "${ETTR_V3_CONFIRMATION_AUDIT}" \
  --main-output-root "${ETTR_V3_MAIN_OUTPUT_ROOT}" \
  --confirmation-output-root "${ETTR_V3_CONFIRMATION_OUTPUT_ROOT}" \
  --report "${ETTR_V3_SEPARATION_REPORT}"
