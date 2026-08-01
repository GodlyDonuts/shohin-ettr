#!/bin/bash
#SBATCH --job-name=shohin-ettr-state-semantic
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc32,evc33,evc40,evc43,evc44
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/state-semantic-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/state-semantic-%j.err

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
DATA_SEED=${DATA_SEED:?set data seed}
UPDATES=${UPDATES:-300}
START_POSITION=${START_POSITION:-13200}
LEARNING_RATE=${LEARNING_RATE:-3e-5}
GRADIENT_CLIP=${GRADIENT_CLIP:-1.0}
COMPILER_AUX_WEIGHT=${COMPILER_AUX_WEIGHT:-0.25}
REACTOR_AUX_WEIGHT=${REACTOR_AUX_WEIGHT:-0.25}
OPTIMIZATION_MODE=${OPTIMIZATION_MODE:-joint-global}
SEMANTIC_PROGRAM_SOURCE=${SEMANTIC_PROGRAM_SOURCE:-predicted}
EVAL_BATCHES=${EVAL_BATCHES:-16}
REACTOR_REDUCTION=${REACTOR_REDUCTION:-head-class-balanced}
REQUIRED_DEVICE_CLASS=${REQUIRED_DEVICE_CLASS:-h100}
PYTHON_ROOT=${PYTHON_ROOT:-/lustre/fs1/home/sa305415/shohin/miniforge3}

for path in "$CODE_ROOT" "$RELEASE_ROOT" "$ETTR_DATA_ROOT" \
  "$TOKENIZER" "$PROTECTED_CHECKPOINT" "$JOINT_RUN_DIR" \
  "$COMPILER_RUN_DIR" "$OUTPUT" "$PYTHON_ROOT"; do
  [[ "$path" == /* ]] || {
    echo "algebraic state-semantic paths must be absolute" >&2
    exit 2
  }
done
cd "$CODE_ROOT"
[[ "$(tr -d '\r\n' < SOURCE_COMMIT)" == "$SOURCE_COMMIT" ]] \
  && [[ "$(sha256sum SHA256SUMS | cut -d ' ' -f 1)" \
    == "$RUNTIME_SHA256SUMS_SHA256" ]] \
  && [[ -z "$(sha256sum -c SHA256SUMS 2>&1 | grep -v ': OK$')" ]] \
  || {
    echo "algebraic state-semantic runtime differs" >&2
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
  echo "algebraic state-semantic output exists" >&2
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
  "$CODE_ROOT/train/train_algebraic_state_semantic_pilot.py" \
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
  --data-seed "$DATA_SEED" \
  --updates "$UPDATES" \
  --start-position "$START_POSITION" \
  --learning-rate "$LEARNING_RATE" \
  --gradient-clip "$GRADIENT_CLIP" \
  --compiler-aux-weight "$COMPILER_AUX_WEIGHT" \
  --reactor-aux-weight "$REACTOR_AUX_WEIGHT" \
  --optimization-mode "$OPTIMIZATION_MODE" \
  --semantic-program-source "$SEMANTIC_PROGRAM_SOURCE" \
  --eval-batches "$EVAL_BATCHES" \
  --reactor-reduction "$REACTOR_REDUCTION" \
  --required-device-class "$REQUIRED_DEVICE_CLASS"

(
  cd "$OUTPUT"
  sha256sum -c SHA256SUMS
)
