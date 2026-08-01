#!/bin/bash
#SBATCH --job-name=shohin-ettr-typed-query
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc32,evc33,evc40,evc43,evc44
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/typed-query-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/typed-query-%j.err

set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set immutable source root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set exact source commit}
RUNTIME_SHA256SUMS_SHA256=${RUNTIME_SHA256SUMS_SHA256:?set runtime receipt hash}
RELEASE_ROOT=${RELEASE_ROOT:?set ETTR release root}
RELEASE_SHA256=${RELEASE_SHA256:?set ETTR release hash}
ETTR_DATA_ROOT=${ETTR_DATA_ROOT:?set ETTR data root}
TOKENIZER=${TOKENIZER:?set source tokenizer path}
TARGET_TOKENIZER=${TARGET_TOKENIZER:?set target tokenizer path}
PARENT_RUN_DIR=${PARENT_RUN_DIR:?set external parent run}
OUTPUT=${OUTPUT:?set fresh pilot output}
DATA_SEED=${DATA_SEED:?set data seed}
MODEL_SEED=${MODEL_SEED:?set model seed}
UPDATES=${UPDATES:-4000}
START_POSITION=${START_POSITION:-12000}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
GRADIENT_CLIP=${GRADIENT_CLIP:-1.0}
EVAL_BATCHES=${EVAL_BATCHES:-32}
WIDTH=${WIDTH:-256}
QUERY_LAYERS=${QUERY_LAYERS:-3}
STATE_LAYERS=${STATE_LAYERS:-2}
NUM_HEADS=${NUM_HEADS:-8}
TEACHER_DECAY_UPDATES=${TEACHER_DECAY_UPDATES:-2000}
PYTHON_ROOT=${PYTHON_ROOT:-/lustre/fs1/home/sa305415/shohin/miniforge3}

for path in "$CODE_ROOT" "$RELEASE_ROOT" "$ETTR_DATA_ROOT" \
  "$TOKENIZER" "$TARGET_TOKENIZER" "$PARENT_RUN_DIR" "$OUTPUT" \
  "$PYTHON_ROOT"; do
  [[ "$path" == /* ]] || {
    echo "typed query-reader paths must be absolute" >&2
    exit 2
  }
done
cd "$CODE_ROOT"
[[ "$(tr -d '\r\n' < SOURCE_COMMIT)" == "$SOURCE_COMMIT" ]] \
  && [[ "$(sha256sum SHA256SUMS | cut -d ' ' -f 1)" \
    == "$RUNTIME_SHA256SUMS_SHA256" ]] \
  && [[ -z "$(sha256sum -c SHA256SUMS 2>&1 | grep -v ': OK$')" ]] \
  || {
    echo "typed query-reader runtime differs" >&2
    exit 2
  }
(
  cd "$PARENT_RUN_DIR"
  sha256sum -c SHA256SUMS
)
if [[ -e "$OUTPUT" || -L "$OUTPUT" ]]; then
  echo "typed query-reader output exists" >&2
  exit 2
fi

parent_contract_sha=$(sha256sum \
  "$PARENT_RUN_DIR/run-contract.json" | cut -d ' ' -f 1)
parent_model_sha=$(sha256sum \
  "$PARENT_RUN_DIR/joint-model-final.pt" | cut -d ' ' -f 1)

"$PYTHON_ROOT/bin/python" \
  "$CODE_ROOT/train/train_typed_query_state_reader_pilot.py" \
  --release-root "$RELEASE_ROOT" \
  --release-sha256 "$RELEASE_SHA256" \
  --data-root "$ETTR_DATA_ROOT" \
  --tokenizer "$TOKENIZER" \
  --target-tokenizer "$TARGET_TOKENIZER" \
  --parent-joint-model "$PARENT_RUN_DIR/joint-model-final.pt" \
  --parent-joint-model-sha256 "$parent_model_sha" \
  --parent-run-contract "$PARENT_RUN_DIR/run-contract.json" \
  --parent-run-contract-sha256 "$parent_contract_sha" \
  --output "$OUTPUT" \
  --source-commit "$SOURCE_COMMIT" \
  --data-seed "$DATA_SEED" \
  --model-seed "$MODEL_SEED" \
  --updates "$UPDATES" \
  --start-position "$START_POSITION" \
  --learning-rate "$LEARNING_RATE" \
  --gradient-clip "$GRADIENT_CLIP" \
  --eval-batches "$EVAL_BATCHES" \
  --width "$WIDTH" \
  --query-layers "$QUERY_LAYERS" \
  --state-layers "$STATE_LAYERS" \
  --num-heads "$NUM_HEADS" \
  --teacher-decay-updates "$TEACHER_DECAY_UPDATES"

(
  cd "$OUTPUT"
  sha256sum -c SHA256SUMS
)
