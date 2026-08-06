#!/bin/bash
#SBATCH -J train-npw1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/train_diverge_npw1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable NPW1 code root}
SOT1_RESULT=${SOT1_RESULT:?set sealed SOT1 result}
SOT1_RESULT_SHA256=${SOT1_RESULT_SHA256:?set sealed SOT1 result hash}
SOT1_CHECKPOINT=${SOT1_CHECKPOINT:?set SOT1 checkpoint}
SOT1_CHECKPOINT_SHA256=${SOT1_CHECKPOINT_SHA256:?set SOT1 checkpoint hash}
TRAINING=${TRAINING:?set NPW1 training data}
TRAINING_SHA256=${TRAINING_SHA256:?set NPW1 training hash}
OUTPUT=${OUTPUT:?set isolated NPW1 output}
SMOKE=${SMOKE:-0}
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

check_hash "$SOT1_RESULT" "$SOT1_RESULT_SHA256"
check_hash "$SOT1_CHECKPOINT" "$SOT1_CHECKPOINT_SHA256"
check_hash "$TRAINING" "$TRAINING_SHA256"
jq -e '.promotion_gate.passed == true' "$SOT1_RESULT" >/dev/null
test ! -e "$OUTPUT" || { echo "refusing existing NPW1 output: $OUTPUT" >&2; exit 2; }

echo "[train-diverge-npw1] node=${SLURMD_NODENAME:-?} code=$CODE_ROOT smoke=$SMOKE"
nvidia-smi -L
"$PYTHON" -m py_compile \
  "$CODE_ROOT/train/diverge_npw1_data.py" \
  "$CODE_ROOT/train/diverge_npw1_runtime.py" \
  "$CODE_ROOT/train/train_diverge_npw1.py"
"$PYTHON" "$CODE_ROOT/train/test_diverge_npw1_data.py"
"$PYTHON" "$CODE_ROOT/train/test_diverge_npw1_runtime.py"

EXTRA=()
if [[ "$SMOKE" == "1" ]]; then
  EXTRA+=(--smoke)
fi
"$PYTHON" "$CODE_ROOT/train/train_diverge_npw1.py" \
  --sot1-checkpoint "$SOT1_CHECKPOINT" \
  --sot1-checkpoint-sha256 "$SOT1_CHECKPOINT_SHA256" \
  --training "$TRAINING" \
  --training-sha256 "$TRAINING_SHA256" \
  --output "$OUTPUT" \
  --device cuda \
  "${EXTRA[@]}"

chmod -R a-w "$OUTPUT"
echo "[train-diverge-npw1] completed"
