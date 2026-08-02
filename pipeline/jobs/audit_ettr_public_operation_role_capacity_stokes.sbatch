#!/bin/bash
#SBATCH --job-name=ettr-op-role-cap
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --output=/home/%u/shohin/logs/ettr-op-role-cap-%j.out
#SBATCH --error=/home/%u/shohin/logs/ettr-op-role-cap-%j.err

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
EFFECT_CAPACITY_REPORT=${EFFECT_CAPACITY_REPORT:?set exact effect-capacity report}
EFFECT_CAPACITY_REPORT_SHA256=${EFFECT_CAPACITY_REPORT_SHA256:?set effect-capacity report hash}
WORKERS=${WORKERS:-48}
PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python3.13}

for path in "$CODE_ROOT" "$DATA_ROOT" "$TOKENIZER" "$OUTPUT" \
  "$EFFECT_CAPACITY_REPORT"; do
  [[ "$path" == /* ]] || {
    echo "public operation role-capacity audit paths must be absolute" >&2
    exit 2
  }
done
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || {
  echo "public operation role-capacity audit workers differ" >&2
  exit 2
}
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || {
  echo "public operation role-capacity audit output already exists" >&2
  exit 2
}
cd "$CODE_ROOT"
[[ "$(tr -d '\r\n' < SOURCE_COMMIT)" == "$SOURCE_COMMIT" ]] \
  && [[ "$(sha256sum SHA256SUMS | cut -d ' ' -f 1)" \
    == "$RUNTIME_SHA256SUMS_SHA256" ]] \
  && [[ -z "$(sha256sum -c SHA256SUMS 2>&1 | grep -v ': OK$')" ]] \
  || {
    echo "public operation role-capacity audit runtime differs" >&2
    exit 2
  }

[[ -x "$PYTHON" && ! -L "$PYTHON" ]] || {
  echo "public operation role-capacity audit Python runtime differs" >&2
  exit 2
}

PYTHONPATH="$CODE_ROOT/pipeline:$CODE_ROOT/train" \
  "$PYTHON" pipeline/audit_ettr_public_operation_role_capacity.py \
  --data-root "$DATA_ROOT" \
  --tokenizer "$TOKENIZER" \
  --effect-capacity-report "$EFFECT_CAPACITY_REPORT" \
  --effect-capacity-report-sha256 "$EFFECT_CAPACITY_REPORT_SHA256" \
  --output "$OUTPUT" \
  --workers "$WORKERS"

sha256sum "$OUTPUT"
