#!/bin/bash
# Create and independently verify immutable train/document/domain holdouts.
#SBATCH -J shohin-v3-holdout
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 2-00:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/v3_holdout_%j.out
#SBATCH --open-mode=append

set -euo pipefail
set -o noclobber
export PATH=/usr/bin:/bin
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
PY=${PY:-$BASE/miniforge3/bin/python}
CODE_ROOT=${CODE_ROOT:?set an immutable complete v3 holdout runtime}
RUNTIME_SHA256SUMS=${RUNTIME_SHA256SUMS:?set the runtime SHA256SUMS path}
RUNTIME_SHA256SUMS_SHA256=${RUNTIME_SHA256SUMS_SHA256:?set its SHA-256}
SOURCE_DIR=${SOURCE_DIR:?set a final near-residual source}
SOURCE_SELECTION_CODE=${SOURCE_SELECTION_CODE:?set its exact selection code}
SOURCE_SELECTION_CODE_SHA256=${SOURCE_SELECTION_CODE_SHA256:?set its SHA-256}
OUTPUT_DIR=${OUTPUT_DIR:?set a fresh split root}
VERIFICATION_RECEIPT=${VERIFICATION_RECEIPT:?set a fresh receipt outside the split}
HOLDOUT_SEED=${HOLDOUT_SEED:?set the frozen nonempty holdout seed}
DOCUMENT_VALIDATION_BPS=${DOCUMENT_VALIDATION_BPS:-100}
DOMAIN_VALIDATION_BPS=${DOMAIN_VALIDATION_BPS:-100}
SHARD_TOKENS=${SHARD_TOKENS:-100000000}
MATERIALIZER=$CODE_ROOT/pipeline/materialize_v3_holdout_split.py
VERIFIER=$CODE_ROOT/pipeline/verify_v3_holdout_split.py

for path in \
  "$CODE_ROOT" "$RUNTIME_SHA256SUMS" "$SOURCE_DIR" \
  "$SOURCE_SELECTION_CODE" "$OUTPUT_DIR" "$VERIFICATION_RECEIPT"; do
  [[ "$path" == /* ]] || {
    echo "[v3-holdout] paths must be absolute: $path" >&2
    exit 2
  }
done
[[ -n "$HOLDOUT_SEED" && "$HOLDOUT_SEED" != *[[:space:]]* \
  && "$DOCUMENT_VALIDATION_BPS" =~ ^[0-9]+$ \
  && "$DOMAIN_VALIDATION_BPS" =~ ^[0-9]+$ \
  && "$SHARD_TOKENS" =~ ^[1-9][0-9]*$ \
  && "$SOURCE_DIR" != *".partial"* \
  && -d "$CODE_ROOT" && ! -L "$CODE_ROOT" \
  && -f "$RUNTIME_SHA256SUMS" && ! -L "$RUNTIME_SHA256SUMS" \
  && -d "$SOURCE_DIR" && ! -L "$SOURCE_DIR" \
  && -f "$SOURCE_SELECTION_CODE" && ! -L "$SOURCE_SELECTION_CODE" \
  && -f "$MATERIALIZER" && ! -L "$MATERIALIZER" \
  && -f "$VERIFIER" && ! -L "$VERIFIER" \
  && -x "$PY" \
  && ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" \
  && ! -e "$VERIFICATION_RECEIPT" && ! -L "$VERIFICATION_RECEIPT" ]] || {
  echo "[v3-holdout] preflight differs" >&2
  exit 2
}
(( DOCUMENT_VALIDATION_BPS + DOMAIN_VALIDATION_BPS < 10000 )) || {
  echo "[v3-holdout] validation allocation differs" >&2
  exit 2
}
printf '%s  %s\n' \
  "$RUNTIME_SHA256SUMS_SHA256" "$RUNTIME_SHA256SUMS" \
  "$SOURCE_SELECTION_CODE_SHA256" "$SOURCE_SELECTION_CODE" | sha256sum -c -
(cd "$CODE_ROOT" && sha256sum -c "$RUNTIME_SHA256SUMS")

mkdir -p "$(dirname "$OUTPUT_DIR")" "$(dirname "$VERIFICATION_RECEIPT")"
export PYTHONPATH="$CODE_ROOT"
cd "$CODE_ROOT"
"$PY" -u "$MATERIALIZER" \
  --source-dir "$SOURCE_DIR" \
  --source-selection-code "$SOURCE_SELECTION_CODE" \
  --selection-code "$MATERIALIZER" \
  --output-dir "$OUTPUT_DIR" \
  --seed "$HOLDOUT_SEED" \
  --document-validation-bps "$DOCUMENT_VALIDATION_BPS" \
  --domain-validation-bps "$DOMAIN_VALIDATION_BPS" \
  --shard-tokens "$SHARD_TOKENS"
"$PY" -u "$VERIFIER" \
  --output-dir "$OUTPUT_DIR" \
  --source-selection-code "$SOURCE_SELECTION_CODE" \
  --selection-code "$MATERIALIZER" > "$VERIFICATION_RECEIPT"
"$PY" - "$VERIFICATION_RECEIPT" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="ascii"))
if value.get("partition_verified") is not True:
    raise SystemExit("[v3-holdout] independent verification differs")
PY
chmod 0444 "$VERIFICATION_RECEIPT"
find "$OUTPUT_DIR" -type f -exec chmod 0444 {} +
find "$OUTPUT_DIR" -depth -type d -exec chmod 0555 {} +
echo "[v3-holdout] complete; split remains quarantined pending utility gates"
