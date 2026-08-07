#!/bin/bash
#SBATCH -J sve1-aggregate
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH -o /lustre/fs1/home/%u/shohin/logs/diverge_sve1_aggregate_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable SVE1 code root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set frozen SVE1 source commit}
DEVELOPMENT=${DEVELOPMENT:?set passing SVE1 development report}
DEVELOPMENT_SHA256=${DEVELOPMENT_SHA256:?set passing SVE1 development hash}
CONFIRMATION_ROOT=${CONFIRMATION_ROOT:?set SVE1 confirmation result root}
OUTPUT=${OUTPUT:?set isolated SVE1 aggregate output}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
mkdir -p "$BASE/logs" "$(dirname "$OUTPUT")"
(cd "$CODE_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
test ! -e "$OUTPUT"
arguments=()
for seed in 2026080843 2026080844 2026080845 2026080846 2026080847; do
  report=$CONFIRMATION_ROOT/confirmation_${seed}.json
  test -f "$report"
  arguments+=(--confirmation "$report")
done
"$PYTHON" "$CODE_ROOT/train/aggregate_diverge_sve1_confirmation.py" \
  --development "$DEVELOPMENT" --development-sha256 "$DEVELOPMENT_SHA256" \
  "${arguments[@]}" --source-commit "$SOURCE_COMMIT" --output "$OUTPUT"
echo "[diverge-sve1-confirmation] aggregate complete"
