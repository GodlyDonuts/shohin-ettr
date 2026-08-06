#!/bin/bash
#SBATCH -J diverge-iem1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/diverge_iem1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
SOURCE_COMMIT=${SOURCE_COMMIT:?set immutable SOURCE_COMMIT}
TRAIN_OUTPUT=${TRAIN_OUTPUT:?set isolated IEM1 training output directory}
EVAL_OUTPUT=${EVAL_OUTPUT:?set isolated IEM1 evaluation path}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

TOL1_DATA=$BASE/artifacts/reasoning/diverge_tol1/board_r1/train.jsonl
TOL1_SHA256=d8b4af0744d3c4232c1de91989a7b6fd4dd3168f45e35c98f9495add1b52b8ba
EVIDENCE_DATA=$BASE/artifacts/reasoning/diverge_nve1/data_da2adcd_r1/evidence_train.jsonl
EVIDENCE_SHA256=7eb27276332a14dcbf57c651c8823393fa7eb18b4122cff30c07b961db285b35
QUERY_DATA=$BASE/artifacts/reasoning/diverge_iem1/data_058d2ad_r1/query_train.jsonl
QUERY_SHA256=8dc6085f7632b56563416fa75c13ff611344ad53c7c2ca4350c54bbcc17301fa
BOARD=$BASE/artifacts/reasoning/diverge_iem1/data_058d2ad_r1/confirmation_board.jsonl
BOARD_SHA256=4294799cace78330eabdbba369d383feac870329edd00e15ba53d3a24cc9ce8a
NVE1_CHECKPOINT=$BASE/artifacts/reasoning/diverge_nve1/model_b06bae1_r1/checkpoint_0001000.pt
NVE1_CHECKPOINT_SHA256=1610815471c695b0d2d198922dd99369e1f45a5dabc1b1c5d8e986b30fd200ff
PROTECTED_NVE1=$BASE/artifacts/reasoning/diverge_nve1/data_da2adcd_r1/confirmation_board.jsonl
PROTECTED_NVE1_SHA256=23e06c65ed89ce9136d3a504384d5be7996a76cddef089a92698e1b01b990add
TOL3_CHECKPOINT=$BASE/artifacts/reasoning/diverge_tol3/component_212b35a_r1/checkpoint_0000750.pt
TOL3_CHECKPOINT_SHA256=b8b9dfe54b7ab4a31a74739625b8650fa4ee93a41221ab5d82610ebc1c030328
PROTECTED_TOL3=$BASE/artifacts/reasoning/diverge_tol3/confirmation_0dfaf53_r1/confirmation.jsonl
PROTECTED_TOL3_SHA256=36a5fb51f5129294fac4a6ea30cef22c4637d4ae79f633bbee65cec2b5735ed3

export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs" "$(dirname "$EVAL_OUTPUT")"
cd "$BASE"

check_hash() {
  local path=$1
  local expected=$2
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected"
}

check_hash "$TOL1_DATA" "$TOL1_SHA256"
check_hash "$EVIDENCE_DATA" "$EVIDENCE_SHA256"
check_hash "$QUERY_DATA" "$QUERY_SHA256"
check_hash "$BOARD" "$BOARD_SHA256"
check_hash "$NVE1_CHECKPOINT" "$NVE1_CHECKPOINT_SHA256"
check_hash "$PROTECTED_NVE1" "$PROTECTED_NVE1_SHA256"
check_hash "$TOL3_CHECKPOINT" "$TOL3_CHECKPOINT_SHA256"
check_hash "$PROTECTED_TOL3" "$PROTECTED_TOL3_SHA256"
test ! -e "$TRAIN_OUTPUT" || { echo "refusing existing training output: $TRAIN_OUTPUT" >&2; exit 2; }
test ! -e "$EVAL_OUTPUT" || { echo "refusing existing evaluation: $EVAL_OUTPUT" >&2; exit 2; }

echo "[diverge-iem1] node=${SLURMD_NODENAME:-?} code=$CODE_ROOT commit=$SOURCE_COMMIT"
nvidia-smi -L
"$PYTHON" -m py_compile \
  "$CODE_ROOT/train/diverge_iem1_data.py" \
  "$CODE_ROOT/train/diverge_iem1_runtime.py" \
  "$CODE_ROOT/train/train_diverge_iem1.py" \
  "$CODE_ROOT/train/eval_diverge_iem1.py"
"$PYTHON" "$CODE_ROOT/train/test_diverge_iem1_data.py"
"$PYTHON" "$CODE_ROOT/train/test_diverge_iem1_runtime.py"

trap 'test ! -e "$TRAIN_OUTPUT" || chmod -R a-w "$TRAIN_OUTPUT"; test ! -e "$EVAL_OUTPUT" || chmod a-w "$EVAL_OUTPUT"' EXIT
"$PYTHON" "$CODE_ROOT/train/train_diverge_iem1.py" \
  --tol1-data "$TOL1_DATA" \
  --tol1-data-sha256 "$TOL1_SHA256" \
  --evidence-data "$EVIDENCE_DATA" \
  --evidence-data-sha256 "$EVIDENCE_SHA256" \
  --query-data "$QUERY_DATA" \
  --query-data-sha256 "$QUERY_SHA256" \
  --nve1-checkpoint "$NVE1_CHECKPOINT" \
  --nve1-checkpoint-sha256 "$NVE1_CHECKPOINT_SHA256" \
  --output "$TRAIN_OUTPUT" \
  --updates 1000 \
  --batch-size 256 \
  --learning-rate 0.001 \
  --seed 2026080614 \
  --log-interval 25 \
  --device cuda

CHECKPOINT=$TRAIN_OUTPUT/checkpoint_0001000.pt
CHECKPOINT_SHA256=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
"$PYTHON" "$CODE_ROOT/train/eval_diverge_iem1.py" \
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
  --output "$EVAL_OUTPUT" \
  --batch-size 512 \
  --device cuda

echo "[diverge-iem1] completed"
