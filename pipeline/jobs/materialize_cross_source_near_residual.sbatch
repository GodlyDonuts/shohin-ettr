#!/bin/bash
# Apply one final near-duplicate ledger to one named exact residual.
#SBATCH -J shohin-v3-near-residual
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 2-00:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/v3_near_residual_%j.out
#SBATCH --open-mode=append

set -euo pipefail
export PATH=/usr/bin:/bin
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
PY=${PY:-$BASE/miniforge3/bin/python}
CODE_ROOT=${CODE_ROOT:?set an immutable complete v3 residual runtime}
RUNTIME_SHA256SUMS=${RUNTIME_SHA256SUMS:?set the runtime SHA256SUMS path}
RUNTIME_SHA256SUMS_SHA256=${RUNTIME_SHA256SUMS_SHA256:?set its SHA-256}
SOURCE_DIR=${SOURCE_DIR:?set a final exact-residual source}
SOURCE_SELECTION_CODE=${SOURCE_SELECTION_CODE:?set its exact selection code}
SOURCE_SELECTION_CODE_SHA256=${SOURCE_SELECTION_CODE_SHA256:?set its SHA-256}
NEAR_DIR=${NEAR_DIR:?set the immutable near-audit directory}
CORPUS_NAME=${CORPUS_NAME:?set the exact corpus name used by the audit}
OUTPUT_DIR=${OUTPUT_DIR:?set a fresh residual output directory}
SHARD_TOKENS=${SHARD_TOKENS:-100000000}
MATERIALIZER=$CODE_ROOT/pipeline/materialize_cross_source_near_residual.py

for path in \
  "$CODE_ROOT" "$RUNTIME_SHA256SUMS" "$SOURCE_DIR" \
  "$SOURCE_SELECTION_CODE" "$NEAR_DIR" "$OUTPUT_DIR"; do
  [[ "$path" == /* ]] || {
    echo "[v3-near-residual] paths must be absolute: $path" >&2
    exit 2
  }
done
[[ "$CORPUS_NAME" =~ ^[a-z0-9_-]+$ \
  && "$SHARD_TOKENS" =~ ^[1-9][0-9]*$ \
  && "$SOURCE_DIR" != *".partial"* \
  && "$NEAR_DIR" != *".partial"* \
  && -d "$CODE_ROOT" && ! -L "$CODE_ROOT" \
  && -f "$RUNTIME_SHA256SUMS" && ! -L "$RUNTIME_SHA256SUMS" \
  && -d "$SOURCE_DIR" && ! -L "$SOURCE_DIR" \
  && -f "$SOURCE_SELECTION_CODE" && ! -L "$SOURCE_SELECTION_CODE" \
  && -d "$NEAR_DIR" && ! -L "$NEAR_DIR" \
  && -f "$MATERIALIZER" && ! -L "$MATERIALIZER" \
  && -x "$PY" && ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || {
  echo "[v3-near-residual] preflight differs" >&2
  exit 2
}
printf '%s  %s\n' \
  "$RUNTIME_SHA256SUMS_SHA256" "$RUNTIME_SHA256SUMS" \
  "$SOURCE_SELECTION_CODE_SHA256" "$SOURCE_SELECTION_CODE" | sha256sum -c -
(cd "$CODE_ROOT" && sha256sum -c "$RUNTIME_SHA256SUMS")

mkdir -p "$(dirname "$OUTPUT_DIR")"
export PYTHONPATH="$CODE_ROOT"
cd "$CODE_ROOT"
"$PY" -u "$MATERIALIZER" \
  --source-dir "$SOURCE_DIR" \
  --near-dir "$NEAR_DIR" \
  --corpus-name "$CORPUS_NAME" \
  --source-selection-code "$SOURCE_SELECTION_CODE" \
  --selection-code "$MATERIALIZER" \
  --output-dir "$OUTPUT_DIR" \
  --shard-tokens "$SHARD_TOKENS"
[[ -s "$OUTPUT_DIR/manifest.json" \
  && -s "$OUTPUT_DIR/documents.jsonl.zst" ]] || {
  echo "[v3-near-residual] output contract differs" >&2
  exit 3
}
find "$OUTPUT_DIR" -type f -exec chmod 0444 {} +
find "$OUTPUT_DIR" -depth -type d -exec chmod 0555 {} +
echo "[v3-near-residual] complete; output remains quarantined"
