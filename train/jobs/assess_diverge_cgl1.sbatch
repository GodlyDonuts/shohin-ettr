#!/bin/bash
#SBATCH -J assess-cgl1
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:10:00
#SBATCH -o /lustre/fs1/home/%u/shohin/logs/assess_diverge_cgl1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
BASE_CODE_ROOT=${BASE_CODE_ROOT:?set immutable qualified base code root}
OVERLAY_ROOT=${OVERLAY_ROOT:?set immutable CGL1 overlay root}
SHOHIN_RESULT=${SHOHIN_RESULT:?set Shohin development report}
SMOLLM2_RESULT=${SMOLLM2_RESULT:?set SmolLM2 development report}
FLIPPED_RESULT=${FLIPPED_RESULT:?set flipped-control development report}
OUTPUT=${OUTPUT:?set fresh CGL1 assessment path}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

export PYTHONPATH="$OVERLAY_ROOT/train:$BASE_CODE_ROOT/train:$BASE_CODE_ROOT/pipeline"
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs"
cd "$BASE_CODE_ROOT"

for path in "$SHOHIN_RESULT" "$SMOLLM2_RESULT" "$FLIPPED_RESULT"; do
  test -f "$path" || { echo "missing CGL1 arm result: $path" >&2; exit 2; }
done
test ! -e "$OUTPUT" || { echo "refusing existing CGL1 assessment" >&2; exit 2; }

"$PYTHON" "$OVERLAY_ROOT/train/assess_diverge_cgl1.py" \
  --shohin "$SHOHIN_RESULT" --shohin-sha256 "$(sha256sum "$SHOHIN_RESULT" | awk '{print $1}')" \
  --smollm2 "$SMOLLM2_RESULT" --smollm2-sha256 "$(sha256sum "$SMOLLM2_RESULT" | awk '{print $1}')" \
  --flipped "$FLIPPED_RESULT" --flipped-sha256 "$(sha256sum "$FLIPPED_RESULT" | awk '{print $1}')" \
  --output "$OUTPUT"
