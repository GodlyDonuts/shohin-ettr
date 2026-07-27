#!/usr/bin/env bash
#SBATCH --job-name=ettr-v3-pilot
#SBATCH --partition=normal
#SBATCH --account=skattel
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --array=0-53
#SBATCH --no-requeue

set -euo pipefail
umask 077

: "${ETTR_V3_SOURCE_ROOT:?ETTR_V3_SOURCE_ROOT is required}"
: "${ETTR_V3_OUTPUT_ROOT:?ETTR_V3_OUTPUT_ROOT is required}"
: "${ETTR_V3_SOURCE_COMMIT:?ETTR_V3_SOURCE_COMMIT is required}"
: "${ETTR_V3_PROTOCOL_FREEZE:?ETTR_V3_PROTOCOL_FREEZE is required}"

PYTHON_BIN="${ETTR_V3_PYTHON_BIN:-/usr/bin/python3.12}"
LIMIT="${ETTR_V3_PILOT_LIMIT:-24}"
BEAM_WIDTH="${ETTR_V3_BEAM_WIDTH:-64}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
OUT="${ETTR_V3_OUTPUT_ROOT}/cell-${TASK_ID}.json"

test -x "${PYTHON_BIN}"
test -d "${ETTR_V3_SOURCE_ROOT}/pipeline"
mkdir -p "${ETTR_V3_OUTPUT_ROOT}"

export PYTHONPATH="${ETTR_V3_SOURCE_ROOT}/pipeline"
exec "${PYTHON_BIN}" \
  "${ETTR_V3_SOURCE_ROOT}/pipeline/ettr_il_v3_pilot.py" \
  --matrix-index "${TASK_ID}" \
  --limit "${LIMIT}" \
  --beam-width "${BEAM_WIDTH}" \
  --source-root "${ETTR_V3_SOURCE_ROOT}" \
  --source-commit "${ETTR_V3_SOURCE_COMMIT}" \
  --protocol-freeze "${ETTR_V3_PROTOCOL_FREEZE}" \
  --out "${OUT}"
