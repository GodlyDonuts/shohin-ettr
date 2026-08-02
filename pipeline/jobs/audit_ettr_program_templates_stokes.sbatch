#!/bin/bash
#SBATCH --job-name=shohin-ettr-program-audit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/home/sa305415/shohin/logs/ettr-program-audit-%j.out
#SBATCH --error=/home/sa305415/shohin/logs/ettr-program-audit-%j.err

set -euo pipefail
export PATH=/usr/bin:/bin
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
PY=/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python3.13

CODE_ROOT=${CODE_ROOT:?set immutable source root}
DEPENDENCY_ROOT=${DEPENDENCY_ROOT:?set immutable materializer source root}
DATA_ROOT=${DATA_ROOT:?set admitted ETTR data root}
OUTPUT=${OUTPUT:?set fresh audit report path}
SOURCE_SHA256=${SOURCE_SHA256:?set exact audit-script SHA-256}

for path in "$CODE_ROOT" "$DEPENDENCY_ROOT" "$DATA_ROOT" "$OUTPUT"; do
  if [[ "$path" != /* || "$path" == *$'\n'* || "$path" == *$'\r'* ]]; then
    echo "program-audit paths must be absolute and single-line" >&2
    exit 2
  fi
done
[[ -d "$CODE_ROOT" && ! -L "$CODE_ROOT" ]] || {
  echo "program-audit source root differs" >&2
  exit 2
}
[[ -d "$DEPENDENCY_ROOT" && ! -L "$DEPENDENCY_ROOT" ]] || {
  echo "program-audit dependency root differs" >&2
  exit 2
}
[[ -d "$DATA_ROOT" && ! -L "$DATA_ROOT" ]] || {
  echo "program-audit data root differs" >&2
  exit 2
}
[[ -x "$PY" && ! -L "$PY" ]] || {
  echo "program-audit Python runtime differs" >&2
  exit 2
}
[[ ! -e "$OUTPUT" ]] || {
  echo "program-audit output already exists" >&2
  exit 2
}

script="$CODE_ROOT/pipeline/audit_ettr_program_templates.py"
observed=$(sha256sum "$script" | awk '{print $1}')
[[ "$observed" == "$SOURCE_SHA256" ]] || {
  echo "program-audit source hash differs" >&2
  exit 2
}

mkdir -p "$(dirname "$OUTPUT")"
PYTHONPATH="$CODE_ROOT/pipeline:$DEPENDENCY_ROOT/pipeline:$DEPENDENCY_ROOT/train" \
  "$PY" "$script" \
  --data-root "$DATA_ROOT" \
  --output "$OUTPUT" \
  --workers "${SLURM_CPUS_PER_TASK:?missing Slurm CPU geometry}"

sha256sum "$OUTPUT"
