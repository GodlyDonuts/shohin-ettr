#!/bin/bash
#SBATCH -J run-sti1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/run_diverge_sti1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable STI1 code root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set immutable source commit}
CHECKPOINT=${CHECKPOINT:?set immutable RRG1 checkpoint}
CHECKPOINT_SHA256=${CHECKPOINT_SHA256:?set RRG1 checkpoint hash}
DEVELOPMENT_OUTPUT=${DEVELOPMENT_OUTPUT:?set isolated STI1 development result}
CONFIRMATION_OUTPUT=${CONFIRMATION_OUTPUT:?set isolated STI1 confirmation result}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

DEVELOPMENT_DATA=$BASE/artifacts/reasoning/diverge_srp1/data_995a86e_r1/confirmation_board.jsonl
DEVELOPMENT_DATA_SHA256=8d5bf36e5eab0d51328a1fbfeabad676efcbe697fdbe1dd64cbe592820aa0216
CONFIRMATION_DATA=$BASE/artifacts/reasoning/diverge_ccr1/data_ec6d2f3_r1/confirmation_board.jsonl
CONFIRMATION_DATA_SHA256=299237068f436ba33a68487b5300fcd724f8c98bd8bfe6b1916a4ebc7541ebf7
SRP1_CHECKPOINT=$BASE/artifacts/reasoning/diverge_srp1/model_906b6f1_r1/checkpoint_0001000.pt
SRP1_CHECKPOINT_SHA256=0e4365d96d86ea791b64e8bc60b46e4441acfc3d2bc043bd9339da3e4bb286cd
PROTECTED_NVE1=$BASE/artifacts/reasoning/diverge_nve1/result_fbb2f33_r1/evaluation.json
PROTECTED_NVE1_SHA256=2cda05808f0e5230fd285c345a7431b6791a22b3c35c62b227df35c0f622d6c3
PROTECTED_TOL3=$BASE/artifacts/reasoning/diverge_tol3_0dfaf53_r1/confirmation_evaluation_0dfaf53_r1.json
PROTECTED_TOL3_SHA256=2f5b1ca3c08f4b82f8e220713441014178da72a257b8fafb67b0887f28ad5700

export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs"
cd "$BASE"

check_hash() {
  local path=$1
  local expected=$2
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected"
}

check_hash "$CHECKPOINT" "$CHECKPOINT_SHA256"
check_hash "$DEVELOPMENT_DATA" "$DEVELOPMENT_DATA_SHA256"
check_hash "$SRP1_CHECKPOINT" "$SRP1_CHECKPOINT_SHA256"
check_hash "$PROTECTED_NVE1" "$PROTECTED_NVE1_SHA256"
check_hash "$PROTECTED_TOL3" "$PROTECTED_TOL3_SHA256"
test ! -e "$DEVELOPMENT_OUTPUT" || { echo "refusing existing STI1 development output" >&2; exit 2; }
test ! -e "$CONFIRMATION_OUTPUT" || { echo "refusing existing STI1 confirmation output" >&2; exit 2; }

echo "[run-diverge-sti1] node=${SLURMD_NODENAME:-?} code=$CODE_ROOT commit=$SOURCE_COMMIT"
nvidia-smi -L
"$PYTHON" -m py_compile \
  "$CODE_ROOT/train/diverge_sti1_runtime.py" \
  "$CODE_ROOT/train/eval_diverge_sti1.py"
"$PYTHON" "$CODE_ROOT/train/test_diverge_sti1_runtime.py"
"$PYTHON" "$CODE_ROOT/train/test_eval_diverge_rrg1.py"

"$PYTHON" "$CODE_ROOT/train/eval_diverge_sti1.py" \
  --source-commit "$SOURCE_COMMIT" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --data "$DEVELOPMENT_DATA" \
  --data-sha256 "$DEVELOPMENT_DATA_SHA256" \
  --split development \
  --protected-nve1-result "$PROTECTED_NVE1" \
  --protected-nve1-result-sha256 "$PROTECTED_NVE1_SHA256" \
  --protected-tol3-result "$PROTECTED_TOL3" \
  --protected-tol3-result-sha256 "$PROTECTED_TOL3_SHA256" \
  --output "$DEVELOPMENT_OUTPUT" \
  --batch-size 512 \
  --device cuda

DEVELOPMENT_STATUS=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$DEVELOPMENT_OUTPUT")
if [[ "$DEVELOPMENT_STATUS" != pass ]]; then
  echo "[run-diverge-sti1] development failed; confirmation remains unopened"
  exit 3
fi

# The sealed confirmation board is first read only after development admission.
check_hash "$CONFIRMATION_DATA" "$CONFIRMATION_DATA_SHA256"
"$PYTHON" "$CODE_ROOT/train/eval_diverge_sti1.py" \
  --source-commit "$SOURCE_COMMIT" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --srp1-checkpoint "$SRP1_CHECKPOINT" \
  --srp1-checkpoint-sha256 "$SRP1_CHECKPOINT_SHA256" \
  --data "$CONFIRMATION_DATA" \
  --data-sha256 "$CONFIRMATION_DATA_SHA256" \
  --split confirmation \
  --protected-nve1-result "$PROTECTED_NVE1" \
  --protected-nve1-result-sha256 "$PROTECTED_NVE1_SHA256" \
  --protected-tol3-result "$PROTECTED_TOL3" \
  --protected-tol3-result-sha256 "$PROTECTED_TOL3_SHA256" \
  --output "$CONFIRMATION_OUTPUT" \
  --batch-size 512 \
  --device cuda

CONFIRMATION_STATUS=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$CONFIRMATION_OUTPUT")
echo "[run-diverge-sti1] completed confirmation_status=$CONFIRMATION_STATUS"
test "$CONFIRMATION_STATUS" = pass
