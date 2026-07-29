#!/bin/bash
# Audit one immutable v3 corpus for exact optimizer-visible secrets and
# sensitive identifiers. This is CPU-only and produces no document text.
#SBATCH -J shohin-sensitive-audit
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/sensitive_audit_%j.out
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
SOURCE_ROOT=${SOURCE_ROOT:?set the complete immutable audit runtime}
CORPUS=${CORPUS:?set one final immutable v3 corpus}
SELECTION_CODE=${SELECTION_CODE:?set the exact source selection code}
OUT=${OUT:?set a fresh text-free audit output}
AUDIT_CODE=$SOURCE_ROOT/pipeline/audit_v3_sensitive_content.py

for path in \
  "$SOURCE_ROOT" \
  "$CORPUS" \
  "$SELECTION_CODE" \
  "$OUT" \
  "$AUDIT_CODE" \
  "$PY"; do
  if [[ "$path" != /* ]]; then
    echo "[sensitive-audit] all paths must be absolute" >&2
    exit 2
  fi
done
if [[ ! -d "$SOURCE_ROOT" || -L "$SOURCE_ROOT" \
  || ! -s "$SOURCE_ROOT/SHA256SUMS" \
  || ! -d "$CORPUS" || -L "$CORPUS" \
  || ! -f "$SELECTION_CODE" || -L "$SELECTION_CODE" \
  || ! -x "$PY" || ! -f "$AUDIT_CODE" \
  || -e "$OUT" || -L "$OUT" ]]; then
  echo "[sensitive-audit] preflight differs" >&2
  exit 2
fi
cd "$SOURCE_ROOT"
if [[ -n "$(sha256sum -c SHA256SUMS 2>&1 | grep -v ': OK$')" ]]; then
  echo "[sensitive-audit] source runtime differs" >&2
  exit 2
fi
"$PY" - <<'PY'
import numpy
import tokenizers
import zstandard

assert numpy.__version__
assert tokenizers.__version__
assert zstandard.__version__
PY
"$PY" -u "$AUDIT_CODE" \
  --corpus-dir "$CORPUS" \
  --selection-code "$SELECTION_CODE" \
  --output-dir "$OUT"
[[ -s "$OUT/report.json" && -s "$OUT/sensitive_findings.jsonl.zst" ]] || {
  echo "[sensitive-audit] final receipt differs" >&2
  exit 2
}
printf '[sensitive-audit] complete output=%s\n' "$OUT"
