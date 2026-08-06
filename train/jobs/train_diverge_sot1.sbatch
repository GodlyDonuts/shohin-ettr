#!/bin/bash
#SBATCH -J train-sot1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/train_diverge_sot1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable SOT1 code root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set immutable source commit}
OUTPUT=${OUTPUT:?set isolated SOT1 model output}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

QUERY_DATA=$BASE/artifacts/reasoning/diverge_iem1/data_058d2ad_r1/query_train.jsonl
QUERY_DATA_SHA256=8dc6085f7632b56563416fa75c13ff611344ad53c7c2ca4350c54bbcc17301fa
SOURCE_CHECKPOINT=$BASE/artifacts/reasoning/diverge_tol3_0dfaf53_r1/component_212b35a_r1/checkpoint_0000750.pt
SOURCE_CHECKPOINT_SHA256=b8b9dfe54b7ab4a31a74739625b8650fa4ee93a41221ab5d82610ebc1c030328
EVIDENCE_CHECKPOINT=$BASE/artifacts/reasoning/diverge_nve1/model_b06bae1_r1/checkpoint_0001000.pt
EVIDENCE_CHECKPOINT_SHA256=1610815471c695b0d2d198922dd99369e1f45a5dabc1b1c5d8e986b30fd200ff

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

check_hash "$QUERY_DATA" "$QUERY_DATA_SHA256"
check_hash "$SOURCE_CHECKPOINT" "$SOURCE_CHECKPOINT_SHA256"
check_hash "$EVIDENCE_CHECKPOINT" "$EVIDENCE_CHECKPOINT_SHA256"
test ! -e "$OUTPUT" || { echo "refusing existing SOT1 output: $OUTPUT" >&2; exit 2; }

echo "[train-diverge-sot1] node=${SLURMD_NODENAME:-?} code=$CODE_ROOT commit=$SOURCE_COMMIT"
nvidia-smi -L
"$PYTHON" -m py_compile \
  "$CODE_ROOT/train/diverge_sot1_data.py" \
  "$CODE_ROOT/train/diverge_sot1_runtime.py" \
  "$CODE_ROOT/train/train_diverge_sot1.py"
"$PYTHON" "$CODE_ROOT/train/test_diverge_sot1_data.py"
"$PYTHON" "$CODE_ROOT/train/test_diverge_sot1_runtime.py"

trap 'test ! -e "$OUTPUT" || chmod -R a-w "$OUTPUT"' EXIT
"$PYTHON" "$CODE_ROOT/train/train_diverge_sot1.py" \
  --query-data "$QUERY_DATA" \
  --query-data-sha256 "$QUERY_DATA_SHA256" \
  --source-checkpoint "$SOURCE_CHECKPOINT" \
  --source-checkpoint-sha256 "$SOURCE_CHECKPOINT_SHA256" \
  --evidence-checkpoint "$EVIDENCE_CHECKPOINT" \
  --evidence-checkpoint-sha256 "$EVIDENCE_CHECKPOINT_SHA256" \
  --output "$OUTPUT" \
  --updates 1000 \
  --batch-size 256 \
  --learning-rate 0.003 \
  --seed 2026080616 \
  --device cuda

echo "[train-diverge-sot1] completed"
