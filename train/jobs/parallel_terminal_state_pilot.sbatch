#!/bin/bash
#SBATCH --job-name=shohin-ettr-terminal-state
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc32,evc33,evc40,evc43,evc44
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/terminal-state-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/terminal-state-%j.err

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
CAUSAL_DELTA_WEIGHT=${CAUSAL_DELTA_WEIGHT:?set causal delta weight}
ARCHITECTURE_SEED=${ARCHITECTURE_SEED:?set architecture seed}
DATA_SEED=${DATA_SEED:?set data seed}
UPDATES=${UPDATES:-500}
START_POSITION=${START_POSITION:-0}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
GRADIENT_CLIP=${GRADIENT_CLIP:-1.0}
TRAINING_INITIAL_STATE=${TRAINING_INITIAL_STATE:-autonomous}
EVAL_BATCHES=${EVAL_BATCHES:-32}
WIDTH=${WIDTH:-512}
LAYERS=${LAYERS:-4}
NUM_HEADS=${NUM_HEADS:-8}
RELATION_WIDTH=${RELATION_WIDTH:-64}
REQUIRED_DEVICE_CLASS=${REQUIRED_DEVICE_CLASS:-h100}
RESIDUAL_EDITS=${RESIDUAL_EDITS:-0}
ATOMIC_EDITS=${ATOMIC_EDITS:-0}
ATOMIC_ACTION_WEIGHT=${ATOMIC_ACTION_WEIGHT:-1.0}
LEXICAL_COMMAND=${LEXICAL_COMMAND:-0}
TOKEN_NATIVE_COMMAND_MASK=${TOKEN_NATIVE_COMMAND_MASK:-0}
TOKEN_NATIVE_COVER_VERIFIED_COMMAND_MASK=${TOKEN_NATIVE_COVER_VERIFIED_COMMAND_MASK:-0}
TOKEN_NATIVE_OCCURRENCE_COMMAND=${TOKEN_NATIVE_OCCURRENCE_COMMAND:-0}
TOKEN_NATIVE_SYNTAX_GRAPH_COMMAND=${TOKEN_NATIVE_SYNTAX_GRAPH_COMMAND:-0}
TOKEN_NATIVE_DECLARATION_BINDING_COMMAND=${TOKEN_NATIVE_DECLARATION_BINDING_COMMAND:-0}
TOKEN_NATIVE_OPERATION_RECURRENCE_COMMAND=${TOKEN_NATIVE_OPERATION_RECURRENCE_COMMAND:-0}
TOKEN_NATIVE_OPERATION_STATE_COMMAND=${TOKEN_NATIVE_OPERATION_STATE_COMMAND:-0}
FACTORIZED_OPERATION_EFFECT_COMMAND=${FACTORIZED_OPERATION_EFFECT_COMMAND:-0}
PYTHON_ROOT=${PYTHON_ROOT:-/lustre/fs1/home/sa305415/shohin/miniforge3}

for path in "$CODE_ROOT" "$RELEASE_ROOT" "$ETTR_DATA_ROOT" \
  "$TOKENIZER" "$PROTECTED_CHECKPOINT" "$JOINT_RUN_DIR" \
  "$COMPILER_RUN_DIR" "$OUTPUT" "$PYTHON_ROOT"; do
  [[ "$path" == /* ]] || {
    echo "terminal-state paths must be absolute" >&2
    exit 2
  }
done
cd "$CODE_ROOT"
[[ "$(tr -d '\r\n' < SOURCE_COMMIT)" == "$SOURCE_COMMIT" ]] \
  && [[ "$(sha256sum SHA256SUMS | cut -d ' ' -f 1)" \
    == "$RUNTIME_SHA256SUMS_SHA256" ]] \
  && [[ -z "$(sha256sum -c SHA256SUMS 2>&1 | grep -v ': OK$')" ]] \
  || {
    echo "terminal-state runtime differs" >&2
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
  echo "terminal-state output exists" >&2
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

residual_args=()
if [[ "$RESIDUAL_EDITS" == 1 ]]; then
  residual_args+=(--residual-edits)
elif [[ "$RESIDUAL_EDITS" != 0 ]]; then
  echo "terminal-state residual edit setting differs" >&2
  exit 2
fi
if [[ "$ATOMIC_EDITS" == 1 ]]; then
  residual_args+=(--atomic-edits)
elif [[ "$ATOMIC_EDITS" != 0 ]]; then
  echo "terminal-state atomic edit setting differs" >&2
  exit 2
fi
if [[ "$RESIDUAL_EDITS" == 1 && "$ATOMIC_EDITS" == 1 ]]; then
  echo "terminal-state edit architectures conflict" >&2
  exit 2
fi
if [[ "$LEXICAL_COMMAND" == 1 ]]; then
  [[ "$ATOMIC_EDITS" == 1 ]] || {
    echo "terminal-state lexical COMMAND requires atomic edits" >&2
    exit 2
  }
  residual_args+=(--lexical-command)
elif [[ "$LEXICAL_COMMAND" != 0 ]]; then
  echo "terminal-state lexical COMMAND setting differs" >&2
  exit 2
fi
if [[ "$TOKEN_NATIVE_COMMAND_MASK" == 1 ]]; then
  [[ "$ATOMIC_EDITS" == 1 && "$LEXICAL_COMMAND" == 1 ]] || {
    echo "terminal-state syntax routing requires lexical atomic edits" >&2
    exit 2
  }
  residual_args+=(--token-native-command-mask)
elif [[ "$TOKEN_NATIVE_COMMAND_MASK" != 0 ]]; then
  echo "terminal-state syntax routing setting differs" >&2
  exit 2
fi
if [[ "$TOKEN_NATIVE_COVER_VERIFIED_COMMAND_MASK" == 1 ]]; then
  [[ "$TOKEN_NATIVE_COMMAND_MASK" == 1 ]] || {
    echo "cover verification requires terminal-state syntax routing" >&2
    exit 2
  }
  residual_args+=(--cover-verified-command-mask)
elif [[ "$TOKEN_NATIVE_COVER_VERIFIED_COMMAND_MASK" != 0 ]]; then
  echo "terminal-state cover verification setting differs" >&2
  exit 2
fi
if [[ "$TOKEN_NATIVE_OCCURRENCE_COMMAND" == 1 ]]; then
  [[ "$TOKEN_NATIVE_COMMAND_MASK" == 1 ]] || {
    echo "terminal-state occurrence linking requires syntax routing" >&2
    exit 2
  }
  residual_args+=(--token-native-occurrence-command)
elif [[ "$TOKEN_NATIVE_OCCURRENCE_COMMAND" != 0 ]]; then
  echo "terminal-state occurrence-link setting differs" >&2
  exit 2
fi
if [[ "$TOKEN_NATIVE_SYNTAX_GRAPH_COMMAND" == 1 ]]; then
  [[ "$TOKEN_NATIVE_COMMAND_MASK" == 1 \
    && "$TOKEN_NATIVE_OCCURRENCE_COMMAND" == 0 ]] || {
    echo "terminal-state syntax graph setting differs" >&2
    exit 2
  }
  residual_args+=(--token-native-syntax-graph-command)
elif [[ "$TOKEN_NATIVE_SYNTAX_GRAPH_COMMAND" != 0 ]]; then
  echo "terminal-state syntax graph setting differs" >&2
  exit 2
fi
if [[ "$TOKEN_NATIVE_DECLARATION_BINDING_COMMAND" == 1 ]]; then
  [[ "$TOKEN_NATIVE_SYNTAX_GRAPH_COMMAND" == 1 ]] || {
    echo "terminal-state declaration binding requires syntax graph" >&2
    exit 2
  }
  residual_args+=(--token-native-declaration-binding-command)
elif [[ "$TOKEN_NATIVE_DECLARATION_BINDING_COMMAND" != 0 ]]; then
  echo "terminal-state declaration binding setting differs" >&2
  exit 2
fi
if [[ "$TOKEN_NATIVE_OPERATION_RECURRENCE_COMMAND" == 1 ]]; then
  [[ "$TOKEN_NATIVE_DECLARATION_BINDING_COMMAND" == 1 ]] || {
    echo "operation recurrence requires declaration binding" >&2
    exit 2
  }
  residual_args+=(--token-native-operation-recurrence-command)
elif [[ "$TOKEN_NATIVE_OPERATION_RECURRENCE_COMMAND" != 0 ]]; then
  echo "terminal-state operation recurrence setting differs" >&2
  exit 2
fi
if [[ "$TOKEN_NATIVE_OPERATION_STATE_COMMAND" == 1 ]]; then
  [[ "$TOKEN_NATIVE_OPERATION_RECURRENCE_COMMAND" == 1 \
    && "$TRAINING_INITIAL_STATE" == oracle ]] || {
    echo "operation-state transitions require oracle recurrent training" >&2
    exit 2
  }
  residual_args+=(--token-native-operation-state-command)
elif [[ "$TOKEN_NATIVE_OPERATION_STATE_COMMAND" != 0 ]]; then
  echo "terminal-state operation-state setting differs" >&2
  exit 2
fi
if [[ "$FACTORIZED_OPERATION_EFFECT_COMMAND" == 1 ]]; then
  [[ "$TOKEN_NATIVE_OPERATION_STATE_COMMAND" == 1 ]] || {
    echo "factorized operation effects require operation-state transitions" >&2
    exit 2
  }
  residual_args+=(--factorized-operation-effect-command)
elif [[ "$FACTORIZED_OPERATION_EFFECT_COMMAND" != 0 ]]; then
  echo "terminal-state factorized operation effect setting differs" >&2
  exit 2
fi

"$PYTHON_ROOT/bin/python" \
  "$CODE_ROOT/train/train_parallel_terminal_state_pilot.py" \
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
  --causal-delta-weight "$CAUSAL_DELTA_WEIGHT" \
  --atomic-action-weight "$ATOMIC_ACTION_WEIGHT" \
  --training-initial-state "$TRAINING_INITIAL_STATE" \
  --eval-batches "$EVAL_BATCHES" \
  --width "$WIDTH" \
  --layers "$LAYERS" \
  --num-heads "$NUM_HEADS" \
  --relation-width "$RELATION_WIDTH" \
  "${residual_args[@]}" \
  --required-device-class "$REQUIRED_DEVICE_CLASS"

(
  cd "$OUTPUT"
  sha256sum -c SHA256SUMS
)
