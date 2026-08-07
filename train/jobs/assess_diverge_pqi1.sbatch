#!/bin/bash
#SBATCH -J assess-pqi1
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:10:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/assess_pqi1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable PQI1 code root}
SHOHIN=${SHOHIN:?set Shohin development result}
SMOLLM2=${SMOLLM2:?set SmolLM2 development result}
SHUFFLED=${SHUFFLED:?set shuffled development result}
OUT=${OUT:?set isolated assessment output}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
cd "$BASE"
test ! -e "$OUT" || { echo "refusing existing PQI1 assessment" >&2; exit 2; }
"$PYTHON" "$CODE_ROOT/train/assess_diverge_pqi1.py" \
  --shohin "$SHOHIN" --smollm2 "$SMOLLM2" --shuffled "$SHUFFLED" --output "$OUT"
