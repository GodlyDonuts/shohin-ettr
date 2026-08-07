#!/bin/bash
#SBATCH -J assess-gti1
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:05:00
#SBATCH -o /lustre/fs1/home/%u/shohin/logs/assess_gti1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
BASE_CODE_ROOT=${BASE_CODE_ROOT:?set immutable qualified base code root}
OVERLAY_ROOT=${OVERLAY_ROOT:?set immutable GTI1 overlay root}
SHOHIN=${SHOHIN:?set Shohin development result}
SMOLLM2=${SMOLLM2:?set SmolLM2 development result}
SHUFFLED=${SHUFFLED:?set shuffled-control development result}
OUTPUT=${OUTPUT:?set GTI1 assessment output}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

export PYTHONPATH="$OVERLAY_ROOT/train:$BASE_CODE_ROOT/train${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs"
test ! -e "$OUTPUT" || { echo "refusing existing GTI1 assessment" >&2; exit 2; }

cd "$BASE_CODE_ROOT"
"$PYTHON" "$OVERLAY_ROOT/train/assess_diverge_gti1.py" \
  --shohin "$SHOHIN" \
  --smollm2 "$SMOLLM2" \
  --shuffled "$SHUFFLED" \
  --output "$OUTPUT"
