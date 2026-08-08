#!/bin/bash
#SBATCH --job-name=normalize-lora-metadata
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/normalize_lora_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/normalize_lora_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${RUNTIME_MANIFEST_SHA256:?RUNTIME_MANIFEST_SHA256 is required}"
: "${SOURCE:?SOURCE checkpoint is required}"
: "${SOURCE_SHA256:?SOURCE_SHA256 is required}"
: "${OUTPUT:?OUTPUT checkpoint is required}"
: "${REPORT:?REPORT is required}"

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test -x "$PYTHON"
test -f "$RUNTIME/pipeline/normalize_lora_checkpoint_metadata.py"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$SOURCE" | cut -d' ' -f1)" = "$SOURCE_SHA256"
test ! -e "$OUTPUT"
test ! -e "$REPORT"
mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$REPORT")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PYTHON" "$RUNTIME/pipeline/normalize_lora_checkpoint_metadata.py" \
  --source "$SOURCE" \
  --source-sha256 "$SOURCE_SHA256" \
  --output "$OUTPUT" \
  --report "$REPORT"
chmod a-w "$OUTPUT" "$REPORT"
sha256sum "$OUTPUT" "$REPORT"
