#!/bin/bash
#SBATCH -J compose-pqi1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH --exclude=evc33,evc43
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/compose_pqi1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
BASE_CODE_ROOT=${BASE_CODE_ROOT:?set immutable PQI1 base runtime}
OVERLAY_ROOT=${OVERLAY_ROOT:?set immutable PQI1 composite overlay}
SOURCE_COMMIT=${SOURCE_COMMIT:?set immutable PQI1 composite commit}
MODEL_ROOT=${MODEL_ROOT:?set qualified SmolLM2 PQI1 model root}
DATA=${DATA:?set sealed PQI1 board}
DATA_SHA256=${DATA_SHA256:?set sealed PQI1 board hash}
DIRECT_CONFIRMATION=${DIRECT_CONFIRMATION:?set passing direct confirmation}
DIRECT_CONFIRMATION_SHA256=${DIRECT_CONFIRMATION_SHA256:?set direct confirmation hash}
OUTPUT=${OUTPUT:?set isolated PQI1 composite output}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

STAGE_CHECKPOINT=$BASE/artifacts/reasoning/diverge_rrg1/model_f23d032_r1/checkpoint_0002000.pt
STAGE_CHECKPOINT_SHA256=05c7b6c7ee68e14133231d572b8c66aced5e3a72a6ba25827498560f723336ae
QUERY_CHECKPOINT=$MODEL_ROOT/model/checkpoint.pt
PARENT=$BASE/train/ettr_smollm2_control_parent_a2026072801_7881d8e/joint-model-final.pt
PARENT_SHA256=8196f810a31e0abe7f3bf0eae0a37b103195f109b7a8e962c7b74b5710c98a02
TOKENIZER=$BASE/artifacts/external/smollm2_135m_instruct_83212e1e/tokenizer.json
TOKENIZER_SHA256=9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c
PROTECTED_NVE1=$BASE/artifacts/reasoning/diverge_nve1/result_fbb2f33_r1/evaluation.json
PROTECTED_NVE1_SHA256=2cda05808f0e5230fd285c345a7431b6791a22b3c35c62b227df35c0f622d6c3
PROTECTED_TOL3=$BASE/artifacts/reasoning/diverge_tol3_0dfaf53_r1/confirmation_evaluation_0dfaf53_r1.json
PROTECTED_TOL3_SHA256=2f5b1ca3c08f4b82f8e220713441014178da72a257b8fafb67b0887f28ad5700

export PYTHONPATH="$OVERLAY_ROOT/train:$BASE_CODE_ROOT/train:$BASE_CODE_ROOT/pipeline"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs"
cd "$BASE"

check_hash() {
  local path=$1
  local expected=$2
  test -f "$path" || { echo "missing PQI1 composite input: $path" >&2; exit 2; }
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected" || {
    echo "PQI1 composite input hash differs: $path" >&2
    exit 2
  }
}

check_hash "$STAGE_CHECKPOINT" "$STAGE_CHECKPOINT_SHA256"
check_hash "$DATA" "$DATA_SHA256"
check_hash "$DIRECT_CONFIRMATION" "$DIRECT_CONFIRMATION_SHA256"
check_hash "$PARENT" "$PARENT_SHA256"
check_hash "$TOKENIZER" "$TOKENIZER_SHA256"
check_hash "$PROTECTED_NVE1" "$PROTECTED_NVE1_SHA256"
check_hash "$PROTECTED_TOL3" "$PROTECTED_TOL3_SHA256"
test -f "$QUERY_CHECKPOINT" || { echo "missing PQI1 query checkpoint" >&2; exit 2; }
test ! -e "$OUTPUT" || { echo "refusing existing PQI1 composite output" >&2; exit 2; }

QUERY_CHECKPOINT_SHA256=$(sha256sum "$QUERY_CHECKPOINT" | awk '{print $1}')
echo "[compose-pqi1] node=${SLURMD_NODENAME:-?} commit=$SOURCE_COMMIT"
nvidia-smi -L
"$PYTHON" -m py_compile \
  "$OVERLAY_ROOT/train/diverge_pqi1_composite.py" \
  "$OVERLAY_ROOT/train/eval_diverge_pqi1_composite.py"
"$PYTHON" "$OVERLAY_ROOT/train/test_diverge_pqi1_composite.py"
"$PYTHON" "$OVERLAY_ROOT/train/eval_diverge_pqi1_composite.py" \
  --source-commit "$SOURCE_COMMIT" \
  --stage-checkpoint "$STAGE_CHECKPOINT" \
  --stage-checkpoint-sha256 "$STAGE_CHECKPOINT_SHA256" \
  --query-checkpoint "$QUERY_CHECKPOINT" \
  --query-checkpoint-sha256 "$QUERY_CHECKPOINT_SHA256" \
  --base "$PARENT" --base-sha256 "$PARENT_SHA256" \
  --tokenizer "$TOKENIZER" --tokenizer-sha256 "$TOKENIZER_SHA256" \
  --direct-confirmation "$DIRECT_CONFIRMATION" \
  --direct-confirmation-sha256 "$DIRECT_CONFIRMATION_SHA256" \
  --data "$DATA" --data-sha256 "$DATA_SHA256" \
  --protected-nve1-result "$PROTECTED_NVE1" \
  --protected-nve1-result-sha256 "$PROTECTED_NVE1_SHA256" \
  --protected-tol3-result "$PROTECTED_TOL3" \
  --protected-tol3-result-sha256 "$PROTECTED_TOL3_SHA256" \
  --output "$OUTPUT" --batch-size 128 --device cuda

STATUS=$(
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$OUTPUT"
)
echo "[compose-pqi1] completed status=$STATUS"
test "$STATUS" = pass
