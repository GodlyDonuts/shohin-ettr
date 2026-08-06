#!/bin/bash
#SBATCH -J eval-iem1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/eval_diverge_iem1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable base code root}
PATCH_ROOT=${PATCH_ROOT:?set immutable evaluation patch root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set immutable source commit}
OUTPUT=${OUTPUT:?set isolated IEM1 evaluation path}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

CHECKPOINT=$BASE/artifacts/reasoning/diverge_iem1/model_8fd48c5_r1/checkpoint_0001000.pt
CHECKPOINT_SHA256=c7560eb5c0bb08a8bbe0f15b1127790859fb2a12bf80d5a89f292be30789e84a
NVE1_CHECKPOINT=$BASE/artifacts/reasoning/diverge_nve1/model_b06bae1_r1/checkpoint_0001000.pt
NVE1_CHECKPOINT_SHA256=1610815471c695b0d2d198922dd99369e1f45a5dabc1b1c5d8e986b30fd200ff
BOARD=$BASE/artifacts/reasoning/diverge_iem1/data_058d2ad_r1/confirmation_board.jsonl
BOARD_SHA256=4294799cace78330eabdbba369d383feac870329edd00e15ba53d3a24cc9ce8a
PROTECTED_NVE1=$BASE/artifacts/reasoning/diverge_nve1/data_da2adcd_r1/confirmation_board.jsonl
PROTECTED_NVE1_SHA256=23e06c65ed89ce9136d3a504384d5be7996a76cddef089a92698e1b01b990add
TOL3_CHECKPOINT=$BASE/artifacts/reasoning/diverge_tol3_0dfaf53_r1/component_212b35a_r1/checkpoint_0000750.pt
TOL3_CHECKPOINT_SHA256=b8b9dfe54b7ab4a31a74739625b8650fa4ee93a41221ab5d82610ebc1c030328
PROTECTED_TOL3=$BASE/artifacts/reasoning/diverge_tol3_0dfaf53_r1/confirmation_0dfaf53_r1/confirmation.jsonl
PROTECTED_TOL3_SHA256=36a5fb51f5129294fac4a6ea30cef22c4637d4ae79f633bbee65cec2b5735ed3

export PYTHONPATH="$PATCH_ROOT/train:$CODE_ROOT/train:$CODE_ROOT/pipeline"
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
check_hash "$NVE1_CHECKPOINT" "$NVE1_CHECKPOINT_SHA256"
check_hash "$BOARD" "$BOARD_SHA256"
check_hash "$PROTECTED_NVE1" "$PROTECTED_NVE1_SHA256"
check_hash "$TOL3_CHECKPOINT" "$TOL3_CHECKPOINT_SHA256"
check_hash "$PROTECTED_TOL3" "$PROTECTED_TOL3_SHA256"
test ! -e "$OUTPUT" || { echo "refusing existing evaluation: $OUTPUT" >&2; exit 2; }

echo "[eval-diverge-iem1] node=${SLURMD_NODENAME:-?} patch=$PATCH_ROOT commit=$SOURCE_COMMIT"
nvidia-smi -L
"$PYTHON" -m py_compile \
  "$PATCH_ROOT/train/diverge_iem1_runtime.py" \
  "$PATCH_ROOT/train/eval_diverge_iem1.py"
"$PYTHON" "$PATCH_ROOT/train/test_diverge_iem1_runtime.py"

trap 'test ! -e "$OUTPUT" || chmod a-w "$OUTPUT"' EXIT
"$PYTHON" "$PATCH_ROOT/train/eval_diverge_iem1.py" \
  --iem1-checkpoint "$CHECKPOINT" \
  --iem1-checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --ceiling-evidence-checkpoint "$NVE1_CHECKPOINT" \
  --ceiling-evidence-checkpoint-sha256 "$NVE1_CHECKPOINT_SHA256" \
  --ceiling-tol3-checkpoint "$TOL3_CHECKPOINT" \
  --ceiling-tol3-checkpoint-sha256 "$TOL3_CHECKPOINT_SHA256" \
  --data "$BOARD" \
  --data-sha256 "$BOARD_SHA256" \
  --protected-nve1-data "$PROTECTED_NVE1" \
  --protected-nve1-data-sha256 "$PROTECTED_NVE1_SHA256" \
  --protected-tol3-data "$PROTECTED_TOL3" \
  --protected-tol3-data-sha256 "$PROTECTED_TOL3_SHA256" \
  --output "$OUTPUT" \
  --batch-size 512 \
  --device cuda

echo "[eval-diverge-iem1] completed"
