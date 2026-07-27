#!/usr/bin/env bash
#SBATCH --job-name=ettr-v3-select
#SBATCH --partition=normal
#SBATCH --account=skattel
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --no-requeue

set -euo pipefail
umask 077

: "${ETTR_V3_SOURCE_ROOT:?ETTR_V3_SOURCE_ROOT is required}"
: "${ETTR_V3_MAIN_CANDIDATE_ROOT:?ETTR_V3_MAIN_CANDIDATE_ROOT is required}"
: "${ETTR_V3_CONFIRMATION_CANDIDATE_ROOT:?ETTR_V3_CONFIRMATION_CANDIDATE_ROOT is required}"
: "${ETTR_V3_MAIN_ROOT:?ETTR_V3_MAIN_ROOT is required}"
: "${ETTR_V3_CONFIRMATION_ROOT:?ETTR_V3_CONFIRMATION_ROOT is required}"
: "${ETTR_V3_SELECTION_REPORT:?ETTR_V3_SELECTION_REPORT is required}"
: "${ETTR_V3_SOURCE_COMMIT:?ETTR_V3_SOURCE_COMMIT is required}"
: "${ETTR_V3_SELECTOR_FREEZE:?ETTR_V3_SELECTOR_FREEZE is required}"

PYTHON_BIN="${ETTR_V3_PYTHON_BIN:-/usr/bin/python3.12}"
test -x "${PYTHON_BIN}"
test ! -e "${ETTR_V3_MAIN_ROOT}"
test ! -e "${ETTR_V3_CONFIRMATION_ROOT}"
test ! -e "${ETTR_V3_SELECTION_REPORT}"
mkdir -p \
  "$(dirname "${ETTR_V3_MAIN_ROOT}")" \
  "$(dirname "${ETTR_V3_CONFIRMATION_ROOT}")" \
  "$(dirname "${ETTR_V3_SELECTION_REPORT}")"

export PYTHONPATH="${ETTR_V3_SOURCE_ROOT}/pipeline"
exec "${PYTHON_BIN}" \
  "${ETTR_V3_SOURCE_ROOT}/pipeline/select_ettr_il_v3.py" \
  --main-candidates "${ETTR_V3_MAIN_CANDIDATE_ROOT}" \
  --confirmation-candidates "${ETTR_V3_CONFIRMATION_CANDIDATE_ROOT}" \
  --main-output "${ETTR_V3_MAIN_ROOT}" \
  --confirmation-output "${ETTR_V3_CONFIRMATION_ROOT}" \
  --report "${ETTR_V3_SELECTION_REPORT}" \
  --selector-source-root "${ETTR_V3_SOURCE_ROOT}" \
  --selector-source-commit "${ETTR_V3_SOURCE_COMMIT}" \
  --selector-freeze "${ETTR_V3_SELECTOR_FREEZE}"
