#!/usr/bin/env bash
#SBATCH --job-name=ettr-v3-audit
#SBATCH --partition=normal
#SBATCH --account=skattel
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --no-requeue

set -euo pipefail
umask 077

: "${ETTR_V3_SOURCE_ROOT:?ETTR_V3_SOURCE_ROOT is required}"
: "${ETTR_V3_TASK_MANIFEST:?ETTR_V3_TASK_MANIFEST is required}"
: "${ETTR_V3_OUTPUT_ROOT:?ETTR_V3_OUTPUT_ROOT is required}"
: "${ETTR_V3_REPORTS_ROOT:?ETTR_V3_REPORTS_ROOT is required}"
: "${ETTR_V3_TOKENIZER:?ETTR_V3_TOKENIZER is required}"
: "${ETTR_V3_MATERIALIZER_FREEZE:?ETTR_V3_MATERIALIZER_FREEZE is required}"
: "${ETTR_V3_AUDIT_REPORT:?ETTR_V3_AUDIT_REPORT is required}"
: "${ETTR_V3_PYTHON_BIN:?ETTR_V3_PYTHON_BIN is required}"

export PYTHONPATH="${ETTR_V3_SOURCE_ROOT}/pipeline"
exec "${ETTR_V3_PYTHON_BIN}" \
  "${ETTR_V3_SOURCE_ROOT}/pipeline/materialize_ettr_il_v3_corpus.py" \
  audit \
  --task-manifest "${ETTR_V3_TASK_MANIFEST}" \
  --output-root "${ETTR_V3_OUTPUT_ROOT}" \
  --reports-root "${ETTR_V3_REPORTS_ROOT}" \
  --tokenizer "${ETTR_V3_TOKENIZER}" \
  --materializer-source-root "${ETTR_V3_SOURCE_ROOT}" \
  --materializer-freeze "${ETTR_V3_MATERIALIZER_FREEZE}" \
  --audit-report "${ETTR_V3_AUDIT_REPORT}"
