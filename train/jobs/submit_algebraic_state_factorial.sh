#!/bin/bash
set -Eeuo pipefail
umask 077

SOFT_CODE_ROOT=${SOFT_CODE_ROOT:?set admitted soft-only runtime}
SOFT_SOURCE_COMMIT=${SOFT_SOURCE_COMMIT:?set exact soft-only commit}
SOFT_RUNTIME_SHA256=${SOFT_RUNTIME_SHA256:?set soft-only runtime receipt}
BASIS_CODE_ROOT=${BASIS_CODE_ROOT:?set admitted basis runtime}
BASIS_SOURCE_COMMIT=${BASIS_SOURCE_COMMIT:?set exact basis commit}
BASIS_RUNTIME_SHA256=${BASIS_RUNTIME_SHA256:?set basis runtime receipt}
RELEASE_ROOT=${RELEASE_ROOT:?set ETTR release root}
RELEASE_SHA256=${RELEASE_SHA256:?set ETTR release hash}
ETTR_DATA_ROOT=${ETTR_DATA_ROOT:?set ETTR data root}
TOKENIZER=${TOKENIZER:?set source tokenizer}
PROTECTED_CHECKPOINT=${PROTECTED_CHECKPOINT:?set protected checkpoint}
JOINT_RUN_DIR=${JOINT_RUN_DIR:?set composed Shohin run}
COMPILER_RUN_DIR=${COMPILER_RUN_DIR:?set algebraic query compiler run}
OUTPUT_ROOT=${OUTPUT_ROOT:?set fresh output parent}
SUBMISSION_RECEIPT=${SUBMISSION_RECEIPT:?set fresh submission receipt}
PYTHON_ROOT=${PYTHON_ROOT:-/lustre/fs1/home/sa305415/shohin/miniforge3}
START_POSITION=${START_POSITION:-13200}
EVAL_BATCHES=${EVAL_BATCHES:-32}

for path in "$SOFT_CODE_ROOT" "$BASIS_CODE_ROOT" "$RELEASE_ROOT" \
  "$ETTR_DATA_ROOT" "$TOKENIZER" "$PROTECTED_CHECKPOINT" \
  "$JOINT_RUN_DIR" "$COMPILER_RUN_DIR" "$OUTPUT_ROOT" \
  "$SUBMISSION_RECEIPT" "$PYTHON_ROOT"; do
  [[ "$path" == /* ]] || {
    echo "state-factorial paths must be absolute" >&2
    exit 2
  }
done
for value in "$SOFT_SOURCE_COMMIT" "$BASIS_SOURCE_COMMIT"; do
  [[ "$value" =~ ^[0-9a-f]{40}$ ]] || {
    echo "state-factorial source commit differs" >&2
    exit 2
  }
done
for value in "$SOFT_RUNTIME_SHA256" "$BASIS_RUNTIME_SHA256" \
  "$RELEASE_SHA256"; do
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || {
    echo "state-factorial receipt differs" >&2
    exit 2
  }
done
[[ "$START_POSITION" =~ ^[0-9]+$ && "$EVAL_BATCHES" =~ ^[0-9]+$ ]] \
  && (( EVAL_BATCHES >= 2 )) \
  && [[ -d "$OUTPUT_ROOT" && ! -e "$SUBMISSION_RECEIPT" \
    && ! -L "$SUBMISSION_RECEIPT" ]] \
  || {
    echo "state-factorial launch geometry differs" >&2
    exit 2
  }

for runtime in "$SOFT_CODE_ROOT" "$BASIS_CODE_ROOT"; do
  [[ -f "$runtime/SOURCE_COMMIT" && -f "$runtime/SHA256SUMS" ]] || {
    echo "state-factorial runtime receipt is absent" >&2
    exit 2
  }
done
[[ "$(tr -d '\r\n' < "$SOFT_CODE_ROOT/SOURCE_COMMIT")" \
    == "$SOFT_SOURCE_COMMIT" \
  && "$(sha256sum "$SOFT_CODE_ROOT/SHA256SUMS" | cut -d ' ' -f 1)" \
    == "$SOFT_RUNTIME_SHA256" \
  && "$(tr -d '\r\n' < "$BASIS_CODE_ROOT/SOURCE_COMMIT")" \
    == "$BASIS_SOURCE_COMMIT" \
  && "$(sha256sum "$BASIS_CODE_ROOT/SHA256SUMS" | cut -d ' ' -f 1)" \
    == "$BASIS_RUNTIME_SHA256" ]] || {
  echo "state-factorial runtime identity differs" >&2
  exit 2
}

common_exports() {
  local code_root=$1
  local source_commit=$2
  local runtime_sha=$3
  local output=$4
  local updates=$5
  local learning_rate=$6
  local data_seed=$7
  local basis_weight=$8
  printf '%s' \
    "CODE_ROOT=$code_root" \
    ",SOURCE_COMMIT=$source_commit" \
    ",RUNTIME_SHA256SUMS_SHA256=$runtime_sha" \
    ",RELEASE_ROOT=$RELEASE_ROOT" \
    ",RELEASE_SHA256=$RELEASE_SHA256" \
    ",ETTR_DATA_ROOT=$ETTR_DATA_ROOT" \
    ",TOKENIZER=$TOKENIZER" \
    ",PROTECTED_CHECKPOINT=$PROTECTED_CHECKPOINT" \
    ",JOINT_RUN_DIR=$JOINT_RUN_DIR" \
    ",COMPILER_RUN_DIR=$COMPILER_RUN_DIR" \
    ",OUTPUT=$output" \
    ",DATA_SEED=$data_seed" \
    ",UPDATES=$updates" \
    ",START_POSITION=$START_POSITION" \
    ",LEARNING_RATE=$learning_rate" \
    ",GRADIENT_CLIP=1.0" \
    ",COMPILER_AUX_WEIGHT=0.25" \
    ",REACTOR_AUX_WEIGHT=0.25" \
    ",OPTIMIZATION_MODE=causal-owner-alternating" \
    ",SEMANTIC_PROGRAM_SOURCE=oracle" \
    ",OWNER_STATE_BRIDGE=oracle-factors" \
    ",SEMANTIC_STATE_MODE=soft" \
    ",SEMANTIC_BASIS_WEIGHT=$basis_weight" \
    ",EVAL_BATCHES=$EVAL_BATCHES" \
    ",REACTOR_REDUCTION=head-class-balanced" \
    ",REQUIRED_DEVICE_CLASS=h100" \
    ",PYTHON_ROOT=$PYTHON_ROOT"
}

submitted=()
LAST_JOB_ID=
LAST_SUBMISSION=
cancel_partial() {
  local job
  for job in "${submitted[@]}"; do
    scancel "$job" >/dev/null 2>&1 || true
  done
}
trap cancel_partial ERR INT TERM

submit_arm() {
  local label=$1
  local code_root=$2
  local source_commit=$3
  local runtime_sha=$4
  local output=$5
  local updates=$6
  local learning_rate=$7
  local data_seed=$8
  local basis_weight=$9
  local dependency=${10:-}
  local exports
  local job
  [[ ! -e "$output" && ! -L "$output" ]] || {
    echo "state-factorial output already exists: $output" >&2
    return 2
  }
  exports=$(common_exports \
    "$code_root" "$source_commit" "$runtime_sha" "$output" \
    "$updates" "$learning_rate" "$data_seed" "$basis_weight")
  if [[ -n "$dependency" ]]; then
    job=$(sbatch --parsable \
      --job-name="ettr-$label" \
      --time=08:00:00 \
      --dependency="afterok:$dependency" \
      --export="$exports" \
      "$code_root/train/jobs/algebraic_state_semantic_pilot.sbatch")
  else
    job=$(sbatch --parsable \
      --job-name="ettr-$label" \
      --time=08:00:00 \
      --export="$exports" \
      "$code_root/train/jobs/algebraic_state_semantic_pilot.sbatch")
  fi
  job=${job%%;*}
  [[ "$job" =~ ^[0-9]+$ ]] || {
    echo "state-factorial scheduler receipt differs" >&2
    return 2
  }
  submitted+=("$job")
  LAST_JOB_ID=$job
  LAST_SUBMISSION=$(printf '%s\t%s\t%s' "$label" "$job" "$output")
}

soft_canary_output="$OUTPUT_ROOT/ettr_state_soft_u20_89b41c6_canary"
basis_canary_output="$OUTPUT_ROOT/ettr_state_basis1_u20_3e0aecc_canary"
submit_arm \
  soft-canary "$SOFT_CODE_ROOT" "$SOFT_SOURCE_COMMIT" \
  "$SOFT_RUNTIME_SHA256" "$soft_canary_output" 20 3e-5 \
  2026080110 0.0
soft_canary=$LAST_SUBMISSION
soft_canary_id=$LAST_JOB_ID
submit_arm \
  basis-canary "$BASIS_CODE_ROOT" "$BASIS_SOURCE_COMMIT" \
  "$BASIS_RUNTIME_SHA256" "$basis_canary_output" 20 3e-5 \
  2026080110 1.0
basis_canary=$LAST_SUBMISSION
basis_canary_id=$LAST_JOB_ID

receipt_lines=("$soft_canary" "$basis_canary")
for objective in soft basis1; do
  if [[ "$objective" == soft ]]; then
    code_root=$SOFT_CODE_ROOT
    source_commit=$SOFT_SOURCE_COMMIT
    runtime_sha=$SOFT_RUNTIME_SHA256
    basis_weight=0.0
    dependency=$soft_canary_id
  else
    code_root=$BASIS_CODE_ROOT
    source_commit=$BASIS_SOURCE_COMMIT
    runtime_sha=$BASIS_RUNTIME_SHA256
    basis_weight=1.0
    dependency=$basis_canary_id
  fi
  for learning_rate in 3e-5 1e-5; do
    lr_label=${learning_rate/e-/em}
    for data_seed in 2026080111 2026080112; do
      label="${objective}-${lr_label}-s${data_seed}"
      output="$OUTPUT_ROOT/ettr_state_${objective}_u5000_${lr_label}_s${data_seed}"
      submit_arm \
        "$label" "$code_root" "$source_commit" "$runtime_sha" \
        "$output" 5000 "$learning_rate" "$data_seed" \
        "$basis_weight" "$dependency"
      receipt_lines+=("$LAST_SUBMISSION")
    done
  done
done

trap - ERR INT TERM
set -o noclobber
{
  printf 'schema\tshohin-ettr-algebraic-state-factorial-submission-v1\n'
  printf 'start_position\t%s\n' "$START_POSITION"
  printf 'eval_batches\t%s\n' "$EVAL_BATCHES"
  printf 'label\tjob_id\toutput\n'
  printf '%s\n' "${receipt_lines[@]}"
} > "$SUBMISSION_RECEIPT"
chmod 400 "$SUBMISSION_RECEIPT"
cat "$SUBMISSION_RECEIPT"
