#!/bin/bash
#SBATCH --job-name=cgl1-confirm-board
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/%u/shohin/logs/cgl1-confirm-board-%j.out
#SBATCH --error=/home/%u/shohin/logs/cgl1-confirm-board-%j.err

set -euo pipefail
umask 077
export PATH=/usr/bin:/bin
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1

CODE_ROOT=${CODE_ROOT:?set immutable CGL1 confirmation source root}
PUBLIC=${PUBLIC:?set immutable CGL1 public training data}
PUBLIC_SHA256=${PUBLIC_SHA256:?set exact CGL1 public SHA-256}
DEV_BOARD=${DEV_BOARD:?set immutable development board}
DEV_BOARD_SHA256=${DEV_BOARD_SHA256:?set development board SHA-256}
PQI_BOARD=${PQI_BOARD:?set immutable prior PQI board}
PQI_BOARD_SHA256=${PQI_BOARD_SHA256:?set prior PQI board SHA-256}
OUTPUT=${OUTPUT:?set fresh CGL1 confirmation directory}
RUNTIME_SHA256SUMS_SHA256=${RUNTIME_SHA256SUMS_SHA256:?set runtime receipt hash}
PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python3.13}

cd "$CODE_ROOT"
[[ "$(sha256sum SHA256SUMS | cut -d ' ' -f 1)" == "$RUNTIME_SHA256SUMS_SHA256" ]] \
  && [[ -z "$(sha256sum -c SHA256SUMS 2>&1 | grep -v ': OK$')" ]] \
  || { echo "CGL1 confirmation runtime differs" >&2; exit 2; }
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || {
  echo "CGL1 confirmation output exists" >&2
  exit 2
}

PYTHONPATH="$CODE_ROOT/train" \
  "$PYTHON" pipeline/build_diverge_cgl1_confirmation.py \
  --cgl1-public "$PUBLIC" --cgl1-public-sha256 "$PUBLIC_SHA256" \
  --prior-board "$DEV_BOARD" --prior-board-sha256 "$DEV_BOARD_SHA256" \
  --prior-board "$PQI_BOARD" --prior-board-sha256 "$PQI_BOARD_SHA256" \
  --output "$OUTPUT" --seed 2026080703

sha256sum "$OUTPUT/confirmation_board.jsonl" "$OUTPUT/report.json"
