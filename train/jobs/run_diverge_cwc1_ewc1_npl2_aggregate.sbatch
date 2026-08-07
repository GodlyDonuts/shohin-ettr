#!/bin/bash
#SBATCH -J cwc1-npl2-agg
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/%u/shohin/logs/diverge_cwc1_ewc1_npl2_aggregate_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
BASE_CODE_ROOT=${BASE_CODE_ROOT:?set immutable confirmed NPL2 runtime}
OVERLAY_ROOT=${OVERLAY_ROOT:?set immutable integration overlay}
OUTPUT=${OUTPUT:?set integration aggregate output}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
export PYTHONPATH="$OVERLAY_ROOT/train:$BASE_CODE_ROOT/train:$BASE_CODE_ROOT/pipeline"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs"
(cd "$BASE_CODE_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
(cd "$OVERLAY_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
test ! -e "$OUTPUT"
"$PYTHON" "$OVERLAY_ROOT/train/aggregate_diverge_cwc1_ewc1_npl2_confirmation.py" \
  --result "${RESULT_1:?}" --result "${RESULT_2:?}" --result "${RESULT_3:?}" \
  --result "${RESULT_4:?}" --result "${RESULT_5:?}" \
  --data-report "${DATA_REPORT:?}" --data-report-sha256 "${DATA_REPORT_SHA256:?}" \
  --wrapper-audit "${WRAPPER_AUDIT:?}" \
  --wrapper-audit-sha256 "${WRAPPER_AUDIT_SHA256:?}" \
  --output "$OUTPUT"
test "$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["gate"]["passed"])' "$OUTPUT")" = True
