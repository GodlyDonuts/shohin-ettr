#!/bin/bash
#SBATCH --job-name=shohin-ettr-tri-eval
#SBATCH --partition=normal
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc32,evc33,evc40,evc43,evc44
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/joint-ettr-tri-eval-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/joint-ettr-tri-eval-%j.err

set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set immutable source root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set exact source commit}
RUNTIME_SHA256SUMS_SHA256=${RUNTIME_SHA256SUMS_SHA256:?set runtime receipt hash}
RELEASE_ROOT=${RELEASE_ROOT:?set ETTR release root}
RELEASE_SHA256=${RELEASE_SHA256:?set ETTR release hash}
ETTR_DATA_ROOT=${ETTR_DATA_ROOT:?set ETTR source root}
TOKENIZER=${TOKENIZER:?set tokenizer path}
TARGET_TOKENIZER=${TARGET_TOKENIZER:-}
PROTECTED_CHECKPOINT=${PROTECTED_CHECKPOINT:?set protected checkpoint}
BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT:-}
PARENT_RUN_DIR=${PARENT_RUN_DIR:?set completed parent run}
RUN_DIR=${RUN_DIR:?set completed tri-stream run}
OUTPUT=${OUTPUT:?set fresh evaluation output}
PYTHON_ROOT=${PYTHON_ROOT:-/lustre/fs1/home/sa305415/shohin/miniforge3}
MAX_BATCHES=${MAX_BATCHES:-128}
EVALUATION_SEED=${EVALUATION_SEED:-}

cd "$CODE_ROOT"
[[ "$(tr -d '\r\n' < SOURCE_COMMIT)" == "$SOURCE_COMMIT" ]] \
  && [[ "$(sha256sum SHA256SUMS | awk '{print $1}')" \
    == "$RUNTIME_SHA256SUMS_SHA256" ]] \
  && [[ -z "$(sha256sum -c SHA256SUMS 2>&1 | grep -v ': OK$')" ]] \
  || {
    echo "tri-stream evaluation runtime differs" >&2
    exit 2
  }
for directory in "$PARENT_RUN_DIR" "$RUN_DIR"; do
  (
    cd "$directory"
    sha256sum -c SHA256SUMS
  )
done
if [[ -e "$OUTPUT" || -L "$OUTPUT" ]]; then
  echo "tri-stream evaluation output exists" >&2
  exit 2
fi

parent_contract_sha=$(sha256sum \
  "$PARENT_RUN_DIR/run-contract.json" | awk '{print $1}')
parent_model_sha=$(sha256sum \
  "$PARENT_RUN_DIR/joint-model-final.pt" | awk '{print $1}')
run_contract_sha=$(sha256sum \
  "$RUN_DIR/run-contract.json" | awk '{print $1}')
joint_model_sha=$(sha256sum \
  "$RUN_DIR/joint-model-final.pt" | awk '{print $1}')

eval_args=(
  --release-root "$RELEASE_ROOT"
  --release-sha256 "$RELEASE_SHA256"
  --data-root "$ETTR_DATA_ROOT"
  --tokenizer "$TOKENIZER"
  --protected-checkpoint "$PROTECTED_CHECKPOINT"
  --parent-run-contract "$PARENT_RUN_DIR/run-contract.json"
  --parent-run-contract-sha256 "$parent_contract_sha"
  --parent-joint-model "$PARENT_RUN_DIR/joint-model-final.pt"
  --parent-joint-model-sha256 "$parent_model_sha"
  --run-contract "$RUN_DIR/run-contract.json"
  --run-contract-sha256 "$run_contract_sha"
  --joint-model "$RUN_DIR/joint-model-final.pt"
  --joint-model-sha256 "$joint_model_sha"
  --output "$OUTPUT"
  --source-commit "$SOURCE_COMMIT"
  --max-batches "$MAX_BATCHES"
)
if [[ -n "$EVALUATION_SEED" ]]; then
  eval_args+=(--evaluation-seed "$EVALUATION_SEED")
fi
if [[ -n "$TARGET_TOKENIZER" || -n "$BASE_SOURCE_ROOT" ]]; then
  [[ -n "$TARGET_TOKENIZER" && -n "$BASE_SOURCE_ROOT" ]] || {
    echo "external evaluation arguments differ" >&2
    exit 2
  }
  eval_args+=(
    --target-tokenizer "$TARGET_TOKENIZER"
    --external-base-root "$BASE_SOURCE_ROOT"
  )
fi

"$PYTHON_ROOT/bin/python" \
  "$CODE_ROOT/train/eval_ettr_joint_instruction_model.py" \
  "${eval_args[@]}"
