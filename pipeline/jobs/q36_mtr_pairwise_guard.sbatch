#!/bin/bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36-mtr-pairwise-guard-%j.out
#SBATCH --error=logs/q36-mtr-pairwise-guard-%j.err

set -euo pipefail

for name in PYTHON RUNTIME SCRIPT SCRIPT_SHA256 TRAINING_ROWS DEVELOPMENT_ROWS REUSE_DEVELOPMENT_DECISIONS MODEL_OUTPUT OUTPUT PRODUCTION_OUTPUT DECISIONS REPORT; do
  [[ -n "${!name:-}" ]] || { printf 'q36 pairwise guard missing %s\n' "$name" >&2; exit 2; }
done
[[ -x "$PYTHON" && -f "$SCRIPT" && ! -L "$SCRIPT" ]] || exit 2
[[ "$(sha256sum "$SCRIPT" | awk '{print $1}')" == "$SCRIPT_SHA256" ]] || exit 2
for path in "$TRAINING_ROWS" "$DEVELOPMENT_ROWS" "$REUSE_DEVELOPMENT_DECISIONS"; do
  [[ -f "$path" && ! -L "$path" ]] || exit 2
done
for path in "$MODEL_OUTPUT" "$OUTPUT" "$PRODUCTION_OUTPUT" "$DECISIONS" "$REPORT"; do
  [[ ! -e "$path" && ! -L "$path" ]] || exit 2
done
export PYTHONPATH="$RUNTIME/pipeline:$RUNTIME/train"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PYTHON" -P -s -B "$SCRIPT" \
  --training-rows "$TRAINING_ROWS" \
  --development-rows "$DEVELOPMENT_ROWS" \
  --reuse-development-decisions "$REUSE_DEVELOPMENT_DECISIONS" \
  --model-output "$MODEL_OUTPUT" \
  --output "$OUTPUT" \
  --production-output "$PRODUCTION_OUTPUT" \
  --decisions "$DECISIONS" \
  --report "$REPORT"
chmod a-w "$MODEL_OUTPUT" "$OUTPUT" "$PRODUCTION_OUTPUT" "$DECISIONS" "$REPORT"
