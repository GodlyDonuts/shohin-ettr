#!/bin/bash
#SBATCH --job-name=ettr-mask-audit
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH --output=/home/%u/shohin/logs/ettr-mask-audit-%j.out
#SBATCH --error=/home/%u/shohin/logs/ettr-mask-audit-%j.err

set -euo pipefail
umask 077
export PATH=/usr/bin:/bin
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

CODE_ROOT=${CODE_ROOT:?set immutable source root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set exact source commit}
RUNTIME_SHA256SUMS_SHA256=${RUNTIME_SHA256SUMS_SHA256:?set runtime receipt hash}
DATA_ROOT=${DATA_ROOT:?set immutable ETTR data root}
TOKENIZER=${TOKENIZER:?set tokenizer artifact}
OUTPUT=${OUTPUT:?set fresh report path}
WORKERS=${WORKERS:-48}
PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python3.13}

for path in "$CODE_ROOT" "$DATA_ROOT" "$TOKENIZER" "$OUTPUT"; do
  [[ "$path" == /* ]] || {
    echo "document-mask audit paths must be absolute" >&2
    exit 2
  }
done
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || {
  echo "document-mask audit workers differ" >&2
  exit 2
}
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || {
  echo "document-mask audit output already exists" >&2
  exit 2
}
cd "$CODE_ROOT"
[[ "$(tr -d '\r\n' < SOURCE_COMMIT)" == "$SOURCE_COMMIT" ]] \
  && [[ "$(sha256sum SHA256SUMS | cut -d ' ' -f 1)" \
    == "$RUNTIME_SHA256SUMS_SHA256" ]] \
  && [[ -z "$(sha256sum -c SHA256SUMS 2>&1 | grep -v ': OK$')" ]] \
  || {
    echo "document-mask audit runtime differs" >&2
    exit 2
  }

[[ -x "$PYTHON" && ! -L "$PYTHON" ]] || {
  echo "document-mask audit Python runtime differs" >&2
  exit 2
}

PYTHONPATH="$CODE_ROOT/pipeline:$CODE_ROOT/train" \
  "$PYTHON" pipeline/audit_ettr_document_mask_integrity.py \
  --data-root "$DATA_ROOT" \
  --tokenizer "$TOKENIZER" \
  --output "$OUTPUT" \
  --workers "$WORKERS"

sha256sum "$OUTPUT"
