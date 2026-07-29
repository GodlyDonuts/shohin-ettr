#!/bin/bash
#SBATCH -J shohin-domain-balance
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/domain_balance_%j.out
#SBATCH --open-mode=append
# Materialize one hash-bound domain-balanced candidate. This job never admits
# the result to a training mix and performs no network access.
set -euo pipefail
export PATH=/usr/bin:/bin

BASE=/lustre/fs1/home/sa305415/shohin
PY=${PYTHON:-$BASE/miniforge3/bin/python}
CODE_ROOT=${CODE_ROOT:?set CODE_ROOT to an isolated pinned runtime}
SOURCE_DIR=${SOURCE_DIR:?set SOURCE_DIR to a verified v3 corpus}
POLICY=${POLICY:?set POLICY to an immutable domain-balance policy}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR to a fresh candidate path}
SHARD_TOKENS=${SHARD_TOKENS:-100000000}
SOURCE_SELECTION_CODE=$CODE_ROOT/pipeline/tokenize_shards.py
SELECTION_CODE=$CODE_ROOT/pipeline/materialize_domain_balanced_residual.py
VERIFY_SHARDS=$CODE_ROOT/pipeline/verify_tokenized_shards.py
SOURCE_SELECTION_CODE_SHA256=${SOURCE_SELECTION_CODE_SHA256:?set exact source-selection SHA-256}
SELECTION_CODE_SHA256=${SELECTION_CODE_SHA256:?set exact materializer SHA-256}
VERIFY_SHARDS_SHA256=${VERIFY_SHARDS_SHA256:?set exact verifier SHA-256}
POLICY_SHA256=${POLICY_SHA256:?set exact policy SHA-256}
RUNTIME_SHA256SUMS=${RUNTIME_SHA256SUMS:?set exact runtime SHA256SUMS path}
RUNTIME_SHA256SUMS_SHA256=${RUNTIME_SHA256SUMS_SHA256:?set exact runtime SHA256SUMS SHA-256}

for path in \
  "$CODE_ROOT" "$SOURCE_DIR" "$POLICY" "$OUTPUT_DIR" "$RUNTIME_SHA256SUMS"; do
  [[ "$path" == /* ]] || {
    echo "[domain-balance] paths must be absolute: $path" >&2
    exit 2
  }
done
[[ -d "$CODE_ROOT" && ! -L "$CODE_ROOT" ]] || {
  echo "[domain-balance] isolated code root is unavailable" >&2
  exit 3
}
[[ -d "$SOURCE_DIR" && ! -L "$SOURCE_DIR" ]] || {
  echo "[domain-balance] source corpus is unavailable" >&2
  exit 3
}
[[ -f "$POLICY" && ! -L "$POLICY" ]] || {
  echo "[domain-balance] policy is unavailable" >&2
  exit 3
}
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || {
  echo "[domain-balance] refusing existing output" >&2
  exit 4
}
[[ "$SHARD_TOKENS" =~ ^[1-9][0-9]*$ ]] || {
  echo "[domain-balance] SHARD_TOKENS must be a positive integer" >&2
  exit 2
}
printf '%s  %s\n' \
  "$SOURCE_SELECTION_CODE_SHA256" "$SOURCE_SELECTION_CODE" \
  "$SELECTION_CODE_SHA256" "$SELECTION_CODE" \
  "$VERIFY_SHARDS_SHA256" "$VERIFY_SHARDS" \
  "$POLICY_SHA256" "$POLICY" \
  "$RUNTIME_SHA256SUMS_SHA256" "$RUNTIME_SHA256SUMS" | sha256sum -c -
(cd "$CODE_ROOT" && sha256sum -c "$RUNTIME_SHA256SUMS")

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
export PYTHONPATH=$CODE_ROOT
cd "$CODE_ROOT"
"$PY" -u "$SELECTION_CODE" \
  --source-dir "$SOURCE_DIR" \
  --source-selection-code "$SOURCE_SELECTION_CODE" \
  --policy "$POLICY" \
  --selection-code "$SELECTION_CODE" \
  --output-dir "$OUTPUT_DIR" \
  --shard-tokens "$SHARD_TOKENS"

test -s "$OUTPUT_DIR/manifest.json"
"$PY" -u "$VERIFY_SHARDS" \
  --shard-dir "$OUTPUT_DIR" \
  --selection-code "$SELECTION_CODE" \
  --require-external-inputs
echo "[domain-balance] verified candidate complete; no training admission"
