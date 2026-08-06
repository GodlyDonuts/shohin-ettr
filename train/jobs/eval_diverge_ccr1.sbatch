#!/bin/bash
#SBATCH -J eval-ccr1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/eval_diverge_ccr1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CCR1 code root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set immutable source commit}
CHECKPOINT=${CHECKPOINT:?set CCR1 checkpoint}
CHECKPOINT_SHA256=${CHECKPOINT_SHA256:?set CCR1 checkpoint hash}
DATA=${DATA:?set CCR1 board}
DATA_SHA256=${DATA_SHA256:?set CCR1 board hash}
SPLIT=${SPLIT:?set development or confirmation}
OUTPUT=${OUTPUT:?set isolated CCR1 evaluation output}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
REQUIRE_PASS=${REQUIRE_PASS:-1}

PROTECTED_NVE1=$BASE/artifacts/reasoning/diverge_nve1/result_fbb2f33_r1/evaluation.json
PROTECTED_NVE1_SHA256=2cda05808f0e5230fd285c345a7431b6791a22b3c35c62b227df35c0f622d6c3
PROTECTED_TOL3=$BASE/artifacts/reasoning/diverge_tol3_0dfaf53_r1/confirmation_evaluation_0dfaf53_r1.json
PROTECTED_TOL3_SHA256=2f5b1ca3c08f4b82f8e220713441014178da72a257b8fafb67b0887f28ad5700
SRP1_CHECKPOINT=${SRP1_CHECKPOINT:-}
SRP1_CHECKPOINT_SHA256=${SRP1_CHECKPOINT_SHA256:-}

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
check_hash "$PROTECTED_NVE1" "$PROTECTED_NVE1_SHA256"
check_hash "$PROTECTED_TOL3" "$PROTECTED_TOL3_SHA256"
if [[ "$SPLIT" == confirmation ]]; then
  test -n "$SRP1_CHECKPOINT" && test -n "$SRP1_CHECKPOINT_SHA256"
  check_hash "$SRP1_CHECKPOINT" "$SRP1_CHECKPOINT_SHA256"
fi
test ! -e "$OUTPUT" || { echo "refusing existing CCR1 result: $OUTPUT" >&2; exit 2; }

echo "[eval-diverge-ccr1] split=$SPLIT node=${SLURMD_NODENAME:-?} code=$CODE_ROOT commit=$SOURCE_COMMIT"
nvidia-smi -L
"$PYTHON" -m py_compile \
  "$CODE_ROOT/train/diverge_ccr1_runtime.py" \
  "$CODE_ROOT/train/eval_diverge_ccr1.py"
"$PYTHON" "$CODE_ROOT/train/test_diverge_ccr1_data.py"
"$PYTHON" "$CODE_ROOT/train/test_diverge_ccr1_runtime.py"
"$PYTHON" "$CODE_ROOT/train/test_eval_diverge_ccr1.py"

EXTRA_ARGS=()
if [[ "$SPLIT" == confirmation ]]; then
  EXTRA_ARGS+=(
    --srp1-checkpoint "$SRP1_CHECKPOINT"
    --srp1-checkpoint-sha256 "$SRP1_CHECKPOINT_SHA256"
  )
fi
trap 'test ! -e "$OUTPUT" || chmod a-w "$OUTPUT"' EXIT
"$PYTHON" "$CODE_ROOT/train/eval_diverge_ccr1.py" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --data "$DATA" \
  --data-sha256 "$DATA_SHA256" \
  --split "$SPLIT" \
  --protected-nve1-result "$PROTECTED_NVE1" \
  --protected-nve1-result-sha256 "$PROTECTED_NVE1_SHA256" \
  --protected-tol3-result "$PROTECTED_TOL3" \
  --protected-tol3-result-sha256 "$PROTECTED_TOL3_SHA256" \
  --output "$OUTPUT" \
  --batch-size 512 \
  --device cuda \
  "${EXTRA_ARGS[@]}"

if [[ "$REQUIRE_PASS" == 1 ]]; then
  test "$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$OUTPUT")" = pass
fi
echo "[eval-diverge-ccr1] completed split=$SPLIT"
