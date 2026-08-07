#!/bin/bash
#SBATCH -J confirm-pqi1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH --exclude=evc33,evc43
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/confirm_pqi1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable PQI1 confirmation code root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set immutable PQI1 source commit}
ASSESSMENT=${ASSESSMENT:?set matched development assessment}
ASSESSMENT_SHA256=${ASSESSMENT_SHA256:?set matched development assessment hash}
MODEL_ROOT=${MODEL_ROOT:?set qualified SmolLM2 PQI1 model root}
DATA=${DATA:?set sealed PQI1 confirmation board}
DATA_SHA256=${DATA_SHA256:?set sealed PQI1 confirmation board hash}
OUTPUT=${OUTPUT:?set isolated PQI1 confirmation output}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

PARENT=$BASE/train/ettr_smollm2_control_parent_a2026072801_7881d8e/joint-model-final.pt
PARENT_SHA256=8196f810a31e0abe7f3bf0eae0a37b103195f109b7a8e962c7b74b5710c98a02
TOKENIZER=$BASE/artifacts/external/smollm2_135m_instruct_83212e1e/tokenizer.json
TOKENIZER_SHA256=9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c
CHECKPOINT=$MODEL_ROOT/model/checkpoint.pt

export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
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
  test -f "$path" || { echo "missing PQI1 confirmation input: $path" >&2; exit 2; }
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected" || {
    echo "PQI1 confirmation input hash differs: $path" >&2
    exit 2
  }
}

check_hash "$ASSESSMENT" "$ASSESSMENT_SHA256"
check_hash "$DATA" "$DATA_SHA256"
check_hash "$PARENT" "$PARENT_SHA256"
check_hash "$TOKENIZER" "$TOKENIZER_SHA256"
test -f "$CHECKPOINT" || { echo "missing PQI1 checkpoint" >&2; exit 2; }
test ! -e "$OUTPUT" || { echo "refusing existing PQI1 confirmation" >&2; exit 2; }

ASSESSMENT_STATUS=$(
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$ASSESSMENT"
)
test "$ASSESSMENT_STATUS" = pass || {
  echo "PQI1 development did not admit confirmation" >&2
  exit 3
}

CHECKPOINT_SHA256=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
echo "[confirm-pqi1] node=${SLURMD_NODENAME:-?} commit=$SOURCE_COMMIT"
nvidia-smi -L
"$PYTHON" -m py_compile \
  "$CODE_ROOT/train/diverge_pqi1_runtime.py" \
  "$CODE_ROOT/train/eval_diverge_pqi1.py"
"$PYTHON" "$CODE_ROOT/train/eval_diverge_pqi1.py" \
  --checkpoint "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --base "$PARENT" --base-sha256 "$PARENT_SHA256" \
  --tokenizer "$TOKENIZER" --tokenizer-sha256 "$TOKENIZER_SHA256" \
  --data "$DATA" --data-sha256 "$DATA_SHA256" \
  --board-type confirmation --output "$OUTPUT" --batch-size 128 --device cuda

STATUS=$(
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$OUTPUT"
)
echo "[confirm-pqi1] completed status=$STATUS"
test "$STATUS" = pass
