#!/bin/bash
#SBATCH --job-name=finalize-qpt1
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/finalize_qpt1_%j.out
#SBATCH --error=logs/finalize_qpt1_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${RUNTIME_MANIFEST_SHA256:?RUNTIME_MANIFEST_SHA256 is required}"
: "${BASELINE_PREFIX:?BASELINE_PREFIX is required}"
: "${TREATMENT_PREFIX:?TREATMENT_PREFIX is required}"
: "${BASELINE_TRAINING:?BASELINE_TRAINING is required}"
: "${TREATMENT_TRAINING:?TREATMENT_TRAINING is required}"
: "${OUTPUT:?OUTPUT is required}"

PYTHON=${PYTHON:-python3}
test -x "$(command -v "$PYTHON")"
test -f "$RUNTIME/pipeline/aggregate_qpt1_product_gate.py"
test ! -e "$OUTPUT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = \
  "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)

mkdir -p logs "$(dirname "$OUTPUT")"
export PYTHONPATH="$RUNTIME"
"$PYTHON" "$RUNTIME/pipeline/aggregate_qpt1_product_gate.py" \
  --baseline-prefix "$BASELINE_PREFIX" \
  --treatment-prefix "$TREATMENT_PREFIX" \
  --baseline-training "$BASELINE_TRAINING" \
  --treatment-training "$TREATMENT_TRAINING" \
  --output "$OUTPUT"

test -s "$OUTPUT"
chmod a-w "$OUTPUT"
