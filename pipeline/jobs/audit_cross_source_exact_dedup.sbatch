#!/bin/bash
# Build a text-free exact-duplicate ledger across final verified v3 corpora.
# The order of CORPUS_SPEC_FILE is the immutable retention priority.
# Each line must be: lowercase-name=/absolute/corpus::/absolute/selection.py
#SBATCH -J shohin-v3-exact-audit
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 2-00:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/v3_exact_audit_%j.out
#SBATCH --open-mode=append

set -euo pipefail
export PATH=/usr/bin:/bin
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
PY=${PY:-$BASE/miniforge3/bin/python}
CODE_ROOT=${CODE_ROOT:?set an immutable complete v3 audit runtime}
RUNTIME_SHA256SUMS=${RUNTIME_SHA256SUMS:?set the runtime SHA256SUMS path}
RUNTIME_SHA256SUMS_SHA256=${RUNTIME_SHA256SUMS_SHA256:?set its SHA-256}
CORPUS_SPEC_FILE=${CORPUS_SPEC_FILE:?set an immutable priority-ordered spec file}
CORPUS_SPEC_FILE_SHA256=${CORPUS_SPEC_FILE_SHA256:?set its SHA-256}
OUTPUT_DIR=${OUTPUT_DIR:?set a fresh exact-audit output directory}
AUDITOR=$CODE_ROOT/pipeline/audit_cross_source_exact_dedup.py

for path in \
  "$CODE_ROOT" "$RUNTIME_SHA256SUMS" "$CORPUS_SPEC_FILE" "$OUTPUT_DIR"; do
  [[ "$path" == /* ]] || {
    echo "[v3-exact-audit] paths must be absolute: $path" >&2
    exit 2
  }
done
[[ -d "$CODE_ROOT" && ! -L "$CODE_ROOT" \
  && -f "$RUNTIME_SHA256SUMS" && ! -L "$RUNTIME_SHA256SUMS" \
  && -f "$CORPUS_SPEC_FILE" && ! -L "$CORPUS_SPEC_FILE" \
  && -f "$AUDITOR" && ! -L "$AUDITOR" \
  && -x "$PY" && ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || {
  echo "[v3-exact-audit] preflight differs" >&2
  exit 2
}
printf '%s  %s\n' \
  "$RUNTIME_SHA256SUMS_SHA256" "$RUNTIME_SHA256SUMS" \
  "$CORPUS_SPEC_FILE_SHA256" "$CORPUS_SPEC_FILE" | sha256sum -c -
(cd "$CODE_ROOT" && sha256sum -c "$RUNTIME_SHA256SUMS")

mapfile -t corpus_specs < "$CORPUS_SPEC_FILE"
(( ${#corpus_specs[@]} >= 2 )) || {
  echo "[v3-exact-audit] at least two corpora are required" >&2
  exit 2
}
arguments=()
for spec in "${corpus_specs[@]}"; do
  [[ "$spec" =~ ^[a-z0-9_-]+=/[^[:space:]]+::/[^[:space:]]+$ \
    && "$spec" != *".partial"* ]] || {
    echo "[v3-exact-audit] corpus specification differs" >&2
    exit 2
  }
  arguments+=(--corpus "$spec")
done

mkdir -p "$(dirname "$OUTPUT_DIR")"
export PYTHONPATH="$CODE_ROOT"
cd "$CODE_ROOT"
"$PY" -u "$AUDITOR" \
  "${arguments[@]}" \
  --selection-code "$AUDITOR" \
  --output-dir "$OUTPUT_DIR"
[[ -s "$OUTPUT_DIR/report.json" \
  && -s "$OUTPUT_DIR/exact_duplicate_removals.jsonl.zst" ]] || {
  echo "[v3-exact-audit] output contract differs" >&2
  exit 3
}
find "$OUTPUT_DIR" -type f -exec chmod 0444 {} +
find "$OUTPUT_DIR" -depth -type d -exec chmod 0555 {} +
echo "[v3-exact-audit] complete; no corpus is training-admitted"
