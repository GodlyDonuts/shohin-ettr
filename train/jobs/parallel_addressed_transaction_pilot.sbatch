#!/bin/bash
#SBATCH --job-name=shohin-ettr-parallel-command
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc32,evc33,evc40,evc43,evc44
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/parallel-command-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/parallel-command-%j.err

set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set immutable source root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set exact source commit}
RUNTIME_SHA256SUMS_SHA256=${RUNTIME_SHA256SUMS_SHA256:?set runtime receipt hash}
RELEASE_ROOT=${RELEASE_ROOT:?set ETTR release root}
RELEASE_SHA256=${RELEASE_SHA256:?set ETTR release hash}
ETTR_DATA_ROOT=${ETTR_DATA_ROOT:?set ETTR data root}
TOKENIZER=${TOKENIZER:?set source tokenizer path}
PROTECTED_CHECKPOINT=${PROTECTED_CHECKPOINT:?set protected checkpoint}
JOINT_RUN_DIR=${JOINT_RUN_DIR:?set composed Shohin run}
COMPILER_RUN_DIR=${COMPILER_RUN_DIR:?set completed algebraic compiler run}
OUTPUT=${OUTPUT:?set fresh pilot output}
ARCHITECTURE_SEED=${ARCHITECTURE_SEED:?set architecture seed}
DATA_SEED=${DATA_SEED:?set data seed}
UPDATES=${UPDATES:-500}
START_POSITION=${START_POSITION:-13200}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
GRADIENT_CLIP=${GRADIENT_CLIP:-1.0}
EVAL_BATCHES=${EVAL_BATCHES:-32}
WIDTH=${WIDTH:-384}
LAYERS=${LAYERS:-3}
NUM_HEADS=${NUM_HEADS:-8}
REQUIRED_DEVICE_CLASS=${REQUIRED_DEVICE_CLASS:-h100}
PYTHON_ROOT=${PYTHON_ROOT:-/lustre/fs1/home/sa305415/shohin/miniforge3}

for path in "$CODE_ROOT" "$RELEASE_ROOT" "$ETTR_DATA_ROOT" \
  "$TOKENIZER" "$PROTECTED_CHECKPOINT" "$JOINT_RUN_DIR" \
  "$COMPILER_RUN_DIR" "$OUTPUT" "$PYTHON_ROOT"; do
  [[ "$path" == /* ]] || {
    echo "parallel transaction paths must be absolute" >&2
    exit 2
  }
done
cd "$CODE_ROOT"
[[ "$(tr -d '\r\n' < SOURCE_COMMIT)" == "$SOURCE_COMMIT" ]] \
  && [[ "$(sha256sum SHA256SUMS | cut -d ' ' -f 1)" \
    == "$RUNTIME_SHA256SUMS_SHA256" ]] \
  && [[ -z "$(sha256sum -c SHA256SUMS 2>&1 | grep -v ': OK$')" ]] \
  || {
    echo "parallel transaction runtime differs" >&2
    exit 2
  }
(
  cd "$JOINT_RUN_DIR"
  sha256sum -c SHA256SUMS
)
(
  cd "$COMPILER_RUN_DIR"
  sha256sum -c SHA256SUMS
)
if [[ -e "$OUTPUT" || -L "$OUTPUT" ]]; then
  echo "parallel transaction output exists" >&2
  exit 2
fi

joint_contract_sha=$(sha256sum \
  "$JOINT_RUN_DIR/run-contract.json" | cut -d ' ' -f 1)
joint_model_sha=$(sha256sum \
  "$JOINT_RUN_DIR/joint-model-final.pt" | cut -d ' ' -f 1)
compiler_contract_sha=$(sha256sum \
  "$COMPILER_RUN_DIR/pilot-contract.json" | cut -d ' ' -f 1)
compiler_sha=$(sha256sum \
  "$COMPILER_RUN_DIR/typed-query-reader-final.safetensors" \
  | cut -d ' ' -f 1)

"$PYTHON_ROOT/bin/python" \
  "$CODE_ROOT/train/train_parallel_addressed_transaction_pilot.py" \
  --release-root "$RELEASE_ROOT" \
  --release-sha256 "$RELEASE_SHA256" \
  --data-root "$ETTR_DATA_ROOT" \
  --tokenizer "$TOKENIZER" \
  --protected-checkpoint "$PROTECTED_CHECKPOINT" \
  --joint-model "$JOINT_RUN_DIR/joint-model-final.pt" \
  --joint-model-sha256 "$joint_model_sha" \
  --joint-run-contract "$JOINT_RUN_DIR/run-contract.json" \
  --joint-run-contract-sha256 "$joint_contract_sha" \
  --compiler "$COMPILER_RUN_DIR/typed-query-reader-final.safetensors" \
  --compiler-sha256 "$compiler_sha" \
  --compiler-contract "$COMPILER_RUN_DIR/pilot-contract.json" \
  --compiler-contract-sha256 "$compiler_contract_sha" \
  --output "$OUTPUT" \
  --source-commit "$SOURCE_COMMIT" \
  --architecture-seed "$ARCHITECTURE_SEED" \
  --data-seed "$DATA_SEED" \
  --updates "$UPDATES" \
  --start-position "$START_POSITION" \
  --learning-rate "$LEARNING_RATE" \
  --gradient-clip "$GRADIENT_CLIP" \
  --eval-batches "$EVAL_BATCHES" \
  --width "$WIDTH" \
  --layers "$LAYERS" \
  --num-heads "$NUM_HEADS" \
  --required-device-class "$REQUIRED_DEVICE_CLASS"

(
  cd "$OUTPUT"
  sha256sum -c SHA256SUMS
)
