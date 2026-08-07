#!/bin/bash
#SBATCH -J npl2-confirm-agg
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/%u/shohin/logs/diverge_npl2_aggregate_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable NPL2 confirmation code root}
OUTPUT=${OUTPUT:?set NPL2 aggregate output}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs"
(cd "$CODE_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
test ! -e "$OUTPUT"
"$PYTHON" "$CODE_ROOT/train/test_aggregate_diverge_npl2_confirmation.py"
"$PYTHON" "$CODE_ROOT/train/aggregate_diverge_npl2_confirmation.py" \
  --result "${RESULT_1:?}" --result "${RESULT_2:?}" --result "${RESULT_3:?}" \
  --result "${RESULT_4:?}" --result "${RESULT_5:?}" \
  --data-report "${DATA_REPORT:?}" --data-report-sha256 "${DATA_REPORT_SHA256:?}" \
  --output "$OUTPUT"
