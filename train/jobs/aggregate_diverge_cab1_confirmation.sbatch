#!/bin/bash
#SBATCH -J cab1-aggregate
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH -o /lustre/fs1/home/%u/shohin/logs/diverge_cab1_aggregate_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CAB1 code root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set frozen CAB1 source commit}
DEVELOPMENT=${DEVELOPMENT:?set passing CAB1 development report}
DEVELOPMENT_SHA256=${DEVELOPMENT_SHA256:?set passing CAB1 development hash}
CONFIRMATION_ROOT=${CONFIRMATION_ROOT:?set CAB1 confirmation result root}
OUTPUT=${OUTPUT:?set isolated CAB1 aggregate output}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
mkdir -p "$BASE/logs" "$(dirname "$OUTPUT")"
(cd "$CODE_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
test ! -e "$OUTPUT"
arguments=()
for seed in 2026080823 2026080824 2026080825 2026080826 2026080827; do
  report=$CONFIRMATION_ROOT/confirmation_${seed}.json
  test -f "$report"
  arguments+=(--confirmation "$report")
done
"$PYTHON" "$CODE_ROOT/train/aggregate_diverge_cab1_confirmation.py" \
  --development "$DEVELOPMENT" --development-sha256 "$DEVELOPMENT_SHA256" \
  "${arguments[@]}" --source-commit "$SOURCE_COMMIT" --output "$OUTPUT"
echo "[diverge-cab1-confirmation] aggregate complete"
