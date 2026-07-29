#!/bin/bash
#SBATCH -J shohin-domain-profile
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/domain_profile_%j.out
#SBATCH --open-mode=append
# Produce a text-free domain-concentration profile from a verified v3 corpus.
set -euo pipefail
export PATH=/usr/bin:/bin

BASE=/lustre/fs1/home/sa305415/shohin
PY=${PYTHON:-$BASE/miniforge3/bin/python}
CODE_ROOT=${CODE_ROOT:?set CODE_ROOT to an isolated pinned profile root}
SOURCE_DIR=${SOURCE_DIR:?set SOURCE_DIR to a verified v3 corpus}
OUTPUT=${OUTPUT:?set OUTPUT to a fresh text-free profile}
SOURCE_SELECTION_CODE=$CODE_ROOT/pipeline/tokenize_shards.py
PROFILE=$CODE_ROOT/pipeline/profile_tokenized_corpus_domains.py
PROFILE_SHA256=${PROFILE_SHA256:?set exact domain-profiler SHA-256}
SOURCE_SELECTION_CODE_SHA256=${SOURCE_SELECTION_CODE_SHA256:?set exact source-selection SHA-256}
RUNTIME_SHA256SUMS=${RUNTIME_SHA256SUMS:?set exact runtime SHA256SUMS path}
RUNTIME_SHA256SUMS_SHA256=${RUNTIME_SHA256SUMS_SHA256:?set exact runtime SHA256SUMS SHA-256}

for path in "$CODE_ROOT" "$SOURCE_DIR" "$OUTPUT" "$RUNTIME_SHA256SUMS"; do
  [[ "$path" == /* ]] || {
    echo "[domain-profile] paths must be absolute: $path" >&2
    exit 2
  }
done
[[ -d "$CODE_ROOT" && ! -L "$CODE_ROOT" ]] || {
  echo "[domain-profile] isolated code root is unavailable" >&2
  exit 3
}
[[ -d "$SOURCE_DIR" && ! -L "$SOURCE_DIR" ]] || {
  echo "[domain-profile] source corpus is unavailable" >&2
  exit 3
}
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || {
  echo "[domain-profile] refusing existing output" >&2
  exit 4
}
printf '%s  %s\n' \
  "$PROFILE_SHA256" "$PROFILE" \
  "$SOURCE_SELECTION_CODE_SHA256" "$SOURCE_SELECTION_CODE" \
  "$RUNTIME_SHA256SUMS_SHA256" "$RUNTIME_SHA256SUMS" | sha256sum -c -
(cd "$CODE_ROOT" && sha256sum -c "$RUNTIME_SHA256SUMS")

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
export PYTHONPATH=$CODE_ROOT
cd "$CODE_ROOT"
"$PY" -u "$PROFILE" \
  --source-dir "$SOURCE_DIR" \
  --source-selection-code "$SOURCE_SELECTION_CODE" \
  --output "$OUTPUT"
test -s "$OUTPUT"
sha256sum "$OUTPUT"
echo "[domain-profile] text-free profile complete; no training admission"
