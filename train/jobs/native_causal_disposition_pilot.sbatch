#!/bin/bash
#SBATCH --job-name=shohin-ettr-native-truth
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc32,evc33,evc40,evc43,evc44
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/native-truth-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/native-truth-%j.err

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
INITIAL_READER=${INITIAL_READER:?set exact warm reader}
INITIAL_READER_SHA256=${INITIAL_READER_SHA256:?set warm reader hash}
OUTPUT=${OUTPUT:?set fresh pilot output}
DATA_SEED=${DATA_SEED:?set data seed}
MODEL_SEED=${MODEL_SEED:?set model seed}
UPDATES=${UPDATES:-2000}
START_POSITION=${START_POSITION:-10000}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
GRADIENT_CLIP=${GRADIENT_CLIP:-1.0}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-1}
EVAL_BATCHES=${EVAL_BATCHES:-32}
TRUTH_MOTOR_HIDDEN=${TRUTH_MOTOR_HIDDEN:-0}
TRAIN_READER=${TRAIN_READER:-0}
MOTOR_QUERY_GEOMETRY=${MOTOR_QUERY_GEOMETRY:-stage}
PYTHON_ROOT=${PYTHON_ROOT:-/lustre/fs1/home/sa305415/shohin/miniforge3}

reader_args=()
case "$TRAIN_READER" in
  0) ;;
  1) reader_args+=(--train-reader) ;;
  *)
    echo "TRAIN_READER must be 0 or 1" >&2
    exit 2
    ;;
esac

for path in "$CODE_ROOT" "$RELEASE_ROOT" "$ETTR_DATA_ROOT" \
  "$TOKENIZER" "$TARGET_TOKENIZER" "$PARENT_RUN_DIR" \
  "$INITIAL_READER" "$OUTPUT" "$PYTHON_ROOT"; do
  [[ "$path" == /* ]] || {
    echo "native truth-motor paths must be absolute" >&2
    exit 2
  }
done
cd "$CODE_ROOT"
[[ "$(tr -d '\r\n' < SOURCE_COMMIT)" == "$SOURCE_COMMIT" ]] \
  && [[ "$(sha256sum SHA256SUMS | cut -d ' ' -f 1)" \
    == "$RUNTIME_SHA256SUMS_SHA256" ]] \
  && [[ -z "$(sha256sum -c SHA256SUMS 2>&1 | grep -v ': OK$')" ]] \
  || {
    echo "native truth-motor runtime differs" >&2
    exit 2
  }
(
  cd "$PARENT_RUN_DIR"
  sha256sum -c SHA256SUMS
)
printf '%s  %s\n' "$INITIAL_READER_SHA256" "$INITIAL_READER" \
  | sha256sum -c -
if [[ -e "$OUTPUT" || -L "$OUTPUT" ]]; then
  echo "native truth-motor output exists" >&2
  exit 2
fi

parent_contract_sha=$(sha256sum \
  "$PARENT_RUN_DIR/run-contract.json" | cut -d ' ' -f 1)
parent_model_sha=$(sha256sum \
  "$PARENT_RUN_DIR/joint-model-final.pt" | cut -d ' ' -f 1)

"$PYTHON_ROOT/bin/python" \
  "$CODE_ROOT/train/train_native_causal_disposition_pilot.py" \
  --release-root "$RELEASE_ROOT" \
  --release-sha256 "$RELEASE_SHA256" \
  --data-root "$ETTR_DATA_ROOT" \
  --tokenizer "$TOKENIZER" \
  --target-tokenizer "$TARGET_TOKENIZER" \
  --parent-joint-model "$PARENT_RUN_DIR/joint-model-final.pt" \
  --parent-joint-model-sha256 "$parent_model_sha" \
  --parent-run-contract "$PARENT_RUN_DIR/run-contract.json" \
  --parent-run-contract-sha256 "$parent_contract_sha" \
  --initial-reader "$INITIAL_READER" \
  --initial-reader-sha256 "$INITIAL_READER_SHA256" \
  --output "$OUTPUT" \
  --source-commit "$SOURCE_COMMIT" \
  --data-seed "$DATA_SEED" \
  --model-seed "$MODEL_SEED" \
  --updates "$UPDATES" \
  --start-position "$START_POSITION" \
  --learning-rate "$LEARNING_RATE" \
  --gradient-clip "$GRADIENT_CLIP" \
  --gradient-accumulation "$GRADIENT_ACCUMULATION" \
  --eval-batches "$EVAL_BATCHES" \
  --truth-motor-hidden "$TRUTH_MOTOR_HIDDEN" \
  --motor-query-geometry "$MOTOR_QUERY_GEOMETRY" \
  "${reader_args[@]}"

(
  cd "$OUTPUT"
  sha256sum -c SHA256SUMS
)
