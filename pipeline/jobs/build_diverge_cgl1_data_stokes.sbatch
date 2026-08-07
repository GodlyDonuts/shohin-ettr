#!/bin/bash
#SBATCH --job-name=diverge-cgl1-data
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/%u/shohin/logs/diverge-cgl1-data-%j.out
#SBATCH --error=/home/%u/shohin/logs/diverge-cgl1-data-%j.err

set -euo pipefail
umask 077
export PATH=/usr/bin:/bin
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1

CODE_ROOT=${CODE_ROOT:?set immutable CGL1 source root}
SOURCE=${SOURCE:?set immutable RRG1 QUERY source}
SOURCE_SHA256=${SOURCE_SHA256:?set exact RRG1 QUERY SHA-256}
OUTPUT=${OUTPUT:?set fresh CGL1 output directory}
RUNTIME_SHA256SUMS_SHA256=${RUNTIME_SHA256SUMS_SHA256:?set runtime receipt hash}
PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python3.13}

for path in "$CODE_ROOT" "$SOURCE" "$OUTPUT"; do
  [[ "$path" == /* && "$path" != *$'\n'* && "$path" != *$'\r'* ]] || {
    echo "CGL1 paths must be absolute and single-line" >&2
    exit 2
  }
done
[[ -d "$CODE_ROOT" && ! -L "$CODE_ROOT" ]] || {
  echo "CGL1 source root differs" >&2
  exit 2
}
[[ -f "$SOURCE" && ! -L "$SOURCE" ]] || {
  echo "CGL1 training source differs" >&2
  exit 2
}
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || {
  echo "CGL1 output already exists" >&2
  exit 2
}
[[ -x "$PYTHON" && ! -L "$PYTHON" ]] || {
  echo "CGL1 Python runtime differs" >&2
  exit 2
}

cd "$CODE_ROOT"
[[ "$(sha256sum SHA256SUMS | cut -d ' ' -f 1)" == "$RUNTIME_SHA256SUMS_SHA256" ]] \
  && [[ -z "$(sha256sum -c SHA256SUMS 2>&1 | grep -v ': OK$')" ]] \
  || {
    echo "CGL1 runtime differs" >&2
    exit 2
  }
[[ "$(sha256sum "$SOURCE" | cut -d ' ' -f 1)" == "$SOURCE_SHA256" ]] || {
  echo "CGL1 source hash differs" >&2
  exit 2
}

PYTHONPATH="$CODE_ROOT/train" \
  "$PYTHON" pipeline/build_diverge_cgl1_data.py \
  --source "$SOURCE" \
  --source-sha256 "$SOURCE_SHA256" \
  --output "$OUTPUT"

sha256sum "$OUTPUT/public.jsonl" "$OUTPUT/supervisor.jsonl" "$OUTPUT/report.json"
