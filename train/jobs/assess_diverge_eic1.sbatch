#!/bin/bash
#SBATCH -J eic1-assess
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:10:00
#SBATCH -o /lustre/fs1/home/%u/shohin/logs/eic1-assess-%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
OVERLAY_ROOT=${OVERLAY_ROOT:?set immutable EIC1 assessor root}
RESULTS=${RESULTS:?set immutable EIC1 result root}
OUTPUT=${OUTPUT:?set one EIC1 assessment output}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

export PYTHONPATH="$OVERLAY_ROOT/train:$OVERLAY_ROOT/pipeline"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs"
(cd "$OVERLAY_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
"$PYTHON" "$OVERLAY_ROOT/train/test_assess_diverge_eic1.py"
"$PYTHON" "$OVERLAY_ROOT/train/assess_diverge_eic1.py" \
  --shohin-involution "$RESULTS/shohin_involution" \
  --shohin-duplicate "$RESULTS/shohin_duplicate" \
  --smollm2-involution "$RESULTS/smollm2_involution" \
  --smollm2-duplicate "$RESULTS/smollm2_duplicate" \
  --output "$OUTPUT"
