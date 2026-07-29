#!/bin/bash
# Remove high-confidence sensitive-content findings from one audited v3 corpus.
# This is CPU-only and publishes a fresh, fully reverified residual.
#SBATCH -J shohin-sensitive-residual
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/sensitive_residual_%j.out
#SBATCH --open-mode=append

set -euo pipefail
export PATH=/usr/bin:/bin
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
PY=${PY:-$BASE/miniforge3/bin/python}
SOURCE_ROOT=${SOURCE_ROOT:?set the complete immutable residual runtime}
SOURCE=${SOURCE:?set one final audited v3 corpus}
SOURCE_SELECTION=${SOURCE_SELECTION:?set the exact source selection code}
AUDIT=${AUDIT:?set the immutable sensitive-content audit}
OUT=${OUT:?set a fresh residual output}
SHARD_TOKENS=${SHARD_TOKENS:-100000000}
SELECTION_CODE=$SOURCE_ROOT/pipeline/materialize_sensitive_residual.py

for path in \
  "$SOURCE_ROOT" \
  "$SOURCE" \
  "$SOURCE_SELECTION" \
  "$AUDIT" \
  "$OUT" \
  "$SELECTION_CODE" \
  "$PY"; do
  if [[ "$path" != /* ]]; then
    echo "[sensitive-residual] all paths must be absolute" >&2
    exit 2
  fi
done
if [[ ! -d "$SOURCE_ROOT" || -L "$SOURCE_ROOT" \
  || ! -s "$SOURCE_ROOT/SHA256SUMS" \
  || ! -d "$SOURCE" || -L "$SOURCE" \
  || ! -f "$SOURCE_SELECTION" || -L "$SOURCE_SELECTION" \
  || ! -d "$AUDIT" || -L "$AUDIT" \
  || ! -x "$PY" || ! -f "$SELECTION_CODE" \
  || -e "$OUT" || -L "$OUT" \
  || ! "$SHARD_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[sensitive-residual] preflight differs" >&2
  exit 2
fi
cd "$SOURCE_ROOT"
if [[ -n "$(sha256sum -c SHA256SUMS 2>&1 | grep -v ': OK$')" ]]; then
  echo "[sensitive-residual] source runtime differs" >&2
  exit 2
fi
"$PY" -u "$SELECTION_CODE" \
  --source-dir "$SOURCE" \
  --source-selection-code "$SOURCE_SELECTION" \
  --audit-dir "$AUDIT" \
  --selection-code "$SELECTION_CODE" \
  --output-dir "$OUT" \
  --shard-tokens "$SHARD_TOKENS"
[[ -s "$OUT/manifest.json" && -s "$OUT/document_ledger.jsonl.zst" ]] || {
  echo "[sensitive-residual] final corpus differs" >&2
  exit 2
}
printf '[sensitive-residual] complete output=%s\n' "$OUT"
