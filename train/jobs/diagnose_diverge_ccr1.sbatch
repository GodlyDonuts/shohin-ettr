#!/bin/bash
#SBATCH -J diag-ccr1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:15:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/diagnose_diverge_ccr1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CCR1 diagnostic code root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set immutable source commit}
CHECKPOINT=${CHECKPOINT:?set frozen CCR1 checkpoint}
CHECKPOINT_SHA256=${CHECKPOINT_SHA256:?set frozen CCR1 checkpoint hash}
DATA=${DATA:?set opened development board}
DATA_SHA256=${DATA_SHA256:?set development board hash}
OUTPUT=${OUTPUT:?set isolated diagnostic output}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs" "$(dirname "$OUTPUT")"
cd "$BASE"

check_hash() {
  local path=$1
  local expected=$2
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected"
}

check_hash "$CHECKPOINT" "$CHECKPOINT_SHA256"
check_hash "$DATA" "$DATA_SHA256"
test ! -e "$OUTPUT" || { echo "refusing existing CCR1 diagnostic: $OUTPUT" >&2; exit 2; }
echo "[diagnose-diverge-ccr1] node=${SLURMD_NODENAME:-?} code=$CODE_ROOT commit=$SOURCE_COMMIT"
nvidia-smi -L
"$PYTHON" -m py_compile "$CODE_ROOT/train/diagnose_diverge_ccr1_transfer.py"
trap 'test ! -e "$OUTPUT" || chmod a-w "$OUTPUT"' EXIT
"$PYTHON" "$CODE_ROOT/train/diagnose_diverge_ccr1_transfer.py" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --data "$DATA" \
  --data-sha256 "$DATA_SHA256" \
  --output "$OUTPUT" \
  --batch-size 512 \
  --device cuda

echo "[diagnose-diverge-ccr1] completed"
