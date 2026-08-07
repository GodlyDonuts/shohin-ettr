#!/bin/bash
#SBATCH -J opb1-aggregate
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH -o /lustre/fs1/home/%u/shohin/logs/diverge_opb1_aggregate_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable OPB1 code root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set frozen OPB1 source commit}
DEVELOPMENT=${DEVELOPMENT:?set passing OPB1 development report}
DEVELOPMENT_SHA256=${DEVELOPMENT_SHA256:?set passing OPB1 development hash}
CONFIRMATION_ROOT=${CONFIRMATION_ROOT:?set OPB1 confirmation root}
OUTPUT=${OUTPUT:?set isolated OPB1 aggregate output}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
mkdir -p "$BASE/logs" "$(dirname "$OUTPUT")"
(cd "$CODE_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
test ! -e "$OUTPUT"
arguments=()
for seed in 2026080863 2026080864 2026080865 2026080866 2026080867; do
  report=$CONFIRMATION_ROOT/confirmation_${seed}.json
  test -f "$report"
  arguments+=(--confirmation "$report")
done
"$PYTHON" "$CODE_ROOT/train/aggregate_diverge_opb1_confirmation.py" \
  --development "$DEVELOPMENT" --development-sha256 "$DEVELOPMENT_SHA256" \
  "${arguments[@]}" --source-commit "$SOURCE_COMMIT" --output "$OUTPUT"
echo "[diverge-opb1-confirmation] aggregate complete"
