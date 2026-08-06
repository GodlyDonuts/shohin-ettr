#!/bin/bash
#SBATCH -J eval-srp1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/eval_diverge_srp1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable SRP1 code root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set immutable source commit}
CHECKPOINT=${CHECKPOINT:?set SRP1 checkpoint}
CHECKPOINT_SHA256=${CHECKPOINT_SHA256:?set SRP1 checkpoint hash}
SOT1_CHECKPOINT=${SOT1_CHECKPOINT:?set frozen SOT1 checkpoint}
SOT1_CHECKPOINT_SHA256=${SOT1_CHECKPOINT_SHA256:?set frozen SOT1 checkpoint hash}
DATA=${DATA:?set fresh SRP1 board}
DATA_SHA256=${DATA_SHA256:?set fresh SRP1 board hash}
OUTPUT=${OUTPUT:?set isolated SRP1 evaluation output}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

PROTECTED_NVE1=$BASE/artifacts/reasoning/diverge_nve1/result_fbb2f33_r1/evaluation.json
PROTECTED_NVE1_SHA256=2cda05808f0e5230fd285c345a7431b6791a22b3c35c62b227df35c0f622d6c3
PROTECTED_TOL3=$BASE/artifacts/reasoning/diverge_tol3_0dfaf53_r1/confirmation_evaluation_0dfaf53_r1.json
PROTECTED_TOL3_SHA256=2f5b1ca3c08f4b82f8e220713441014178da72a257b8fafb67b0887f28ad5700

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
check_hash "$SOT1_CHECKPOINT" "$SOT1_CHECKPOINT_SHA256"
check_hash "$DATA" "$DATA_SHA256"
check_hash "$PROTECTED_NVE1" "$PROTECTED_NVE1_SHA256"
check_hash "$PROTECTED_TOL3" "$PROTECTED_TOL3_SHA256"
test ! -e "$OUTPUT" || { echo "refusing existing SRP1 result: $OUTPUT" >&2; exit 2; }

echo "[eval-diverge-srp1] node=${SLURMD_NODENAME:-?} code=$CODE_ROOT commit=$SOURCE_COMMIT"
nvidia-smi -L
"$PYTHON" -m py_compile \
  "$CODE_ROOT/train/diverge_srp1_runtime.py" \
  "$CODE_ROOT/train/eval_diverge_srp1.py"
"$PYTHON" "$CODE_ROOT/train/test_diverge_srp1_data.py"
"$PYTHON" "$CODE_ROOT/train/test_diverge_srp1_runtime.py"

trap 'test ! -e "$OUTPUT" || chmod a-w "$OUTPUT"' EXIT
"$PYTHON" "$CODE_ROOT/train/eval_diverge_srp1.py" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --sot1-checkpoint "$SOT1_CHECKPOINT" \
  --sot1-checkpoint-sha256 "$SOT1_CHECKPOINT_SHA256" \
  --data "$DATA" \
  --data-sha256 "$DATA_SHA256" \
  --protected-nve1-result "$PROTECTED_NVE1" \
  --protected-nve1-result-sha256 "$PROTECTED_NVE1_SHA256" \
  --protected-tol3-result "$PROTECTED_TOL3" \
  --protected-tol3-result-sha256 "$PROTECTED_TOL3_SHA256" \
  --output "$OUTPUT" \
  --batch-size 512 \
  --device cuda

echo "[eval-diverge-srp1] completed"

