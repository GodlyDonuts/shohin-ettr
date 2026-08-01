#!/bin/bash
set -Eeuo pipefail
umask 077

BASIS_CODE_ROOT=${BASIS_CODE_ROOT:?set admitted basis runtime}
BASIS_SOURCE_COMMIT=${BASIS_SOURCE_COMMIT:?set exact basis commit}
BASIS_RUNTIME_SHA256=${BASIS_RUNTIME_SHA256:?set basis runtime receipt}
RUNTIME_TAG=${RUNTIME_TAG:?set immutable runtime tag}
OBJECTIVE_TAG=${OBJECTIVE_TAG:-basis1r2}
SEMANTIC_ANSWER_WEIGHT=${SEMANTIC_ANSWER_WEIGHT:-1.0}
SEMANTIC_BASIS_SCORING=${SEMANTIC_BASIS_SCORING:-log}
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

for path in "$BASIS_CODE_ROOT" "$RELEASE_ROOT" "$ETTR_DATA_ROOT" \
  "$TOKENIZER" "$PROTECTED_CHECKPOINT" "$JOINT_RUN_DIR" \
  "$COMPILER_RUN_DIR" "$OUTPUT_ROOT" "$SUBMISSION_RECEIPT" \
  "$PYTHON_ROOT"; do
  [[ "$path" == /* ]] || {
    echo "basis-recovery paths must be absolute" >&2
    exit 2
  }
done
[[ "$BASIS_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ \
  && "$BASIS_RUNTIME_SHA256" =~ ^[0-9a-f]{64}$ \
  && "$RELEASE_SHA256" =~ ^[0-9a-f]{64}$ \
  && "$RUNTIME_TAG" =~ ^[0-9a-z]+$ \
  && "$OBJECTIVE_TAG" =~ ^[0-9a-z]+$ \
  && "$START_POSITION" =~ ^[0-9]+$ \
  && "$EVAL_BATCHES" =~ ^[0-9]+$ \
  && "$EVAL_BATCHES" -ge 2 \
  && -d "$OUTPUT_ROOT" \
  && ! -e "$SUBMISSION_RECEIPT" \
  && ! -L "$SUBMISSION_RECEIPT" ]] || {
  echo "basis-recovery launch contract differs" >&2
  exit 2
}
case "$OBJECTIVE_TAG:$SEMANTIC_ANSWER_WEIGHT:$SEMANTIC_BASIS_SCORING" in
  basis1r2:1.0:log|qbrier:0.0:brier) ;;
  *)
    echo "basis-recovery objective contract differs" >&2
    exit 2
    ;;
esac
[[ -f "$BASIS_CODE_ROOT/SOURCE_COMMIT" \
  && -f "$BASIS_CODE_ROOT/SHA256SUMS" \
  && "$(tr -d '\r\n' < "$BASIS_CODE_ROOT/SOURCE_COMMIT")" \
    == "$BASIS_SOURCE_COMMIT" \
  && "$(sha256sum "$BASIS_CODE_ROOT/SHA256SUMS" | cut -d ' ' -f 1)" \
    == "$BASIS_RUNTIME_SHA256" ]] || {
  echo "basis-recovery runtime identity differs" >&2
  exit 2
}

exports_for() {
  local output=$1
  local updates=$2
  local learning_rate=$3
  local data_seed=$4
  printf '%s' \
    "CODE_ROOT=$BASIS_CODE_ROOT" \
    ",SOURCE_COMMIT=$BASIS_SOURCE_COMMIT" \
    ",RUNTIME_SHA256SUMS_SHA256=$BASIS_RUNTIME_SHA256" \
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
    ",SEMANTIC_ANSWER_WEIGHT=$SEMANTIC_ANSWER_WEIGHT" \
    ",SEMANTIC_BASIS_WEIGHT=1.0" \
    ",SEMANTIC_BASIS_SCORING=$SEMANTIC_BASIS_SCORING" \
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
  local output=$2
  local updates=$3
  local learning_rate=$4
  local data_seed=$5
  local dependency=${6:-}
  local exports
  local job
  [[ ! -e "$output" && ! -L "$output" ]] || {
    echo "basis-recovery output already exists: $output" >&2
    return 2
  }
  exports=$(exports_for "$output" "$updates" "$learning_rate" "$data_seed")
  if [[ -n "$dependency" ]]; then
    job=$(sbatch --parsable --job-name="ettr-$label" --time=08:00:00 \
      --dependency="afterok:$dependency" --export="$exports" \
      "$BASIS_CODE_ROOT/train/jobs/algebraic_state_semantic_pilot.sbatch")
  else
    job=$(sbatch --parsable --job-name="ettr-$label" --time=08:00:00 \
      --export="$exports" \
      "$BASIS_CODE_ROOT/train/jobs/algebraic_state_semantic_pilot.sbatch")
  fi
  job=${job%%;*}
  [[ "$job" =~ ^[0-9]+$ ]] || {
    echo "basis-recovery scheduler receipt differs" >&2
    return 2
  }
  submitted+=("$job")
  LAST_JOB_ID=$job
  LAST_SUBMISSION=$(printf '%s\t%s\t%s' "$label" "$job" "$output")
}

canary_output="$OUTPUT_ROOT/ettr_state_${OBJECTIVE_TAG}_u20_${RUNTIME_TAG}_canary"
submit_arm "${OBJECTIVE_TAG}-canary" "$canary_output" 20 3e-5 2026080110
canary_id=$LAST_JOB_ID
receipt_lines=("$LAST_SUBMISSION")
for learning_rate in 3e-5 1e-5; do
  lr_label=${learning_rate/e-/em}
  for data_seed in 2026080111 2026080112; do
    label="${OBJECTIVE_TAG}-${lr_label}-s${data_seed}"
    output="$OUTPUT_ROOT/ettr_state_${OBJECTIVE_TAG}_u5000_${lr_label}_s${data_seed}"
    submit_arm "$label" "$output" 5000 "$learning_rate" "$data_seed" \
      "$canary_id"
    receipt_lines+=("$LAST_SUBMISSION")
  done
done

trap - ERR INT TERM
set -o noclobber
{
  printf 'schema\tshohin-ettr-algebraic-state-basis-recovery-v1\n'
  printf 'runtime_tag\t%s\n' "$RUNTIME_TAG"
  printf 'objective_tag\t%s\n' "$OBJECTIVE_TAG"
  printf 'semantic_answer_weight\t%s\n' "$SEMANTIC_ANSWER_WEIGHT"
  printf 'semantic_basis_scoring\t%s\n' "$SEMANTIC_BASIS_SCORING"
  printf 'start_position\t%s\n' "$START_POSITION"
  printf 'eval_batches\t%s\n' "$EVAL_BATCHES"
  printf 'label\tjob_id\toutput\n'
  printf '%s\n' "${receipt_lines[@]}"
} > "$SUBMISSION_RECEIPT"
chmod 400 "$SUBMISSION_RECEIPT"
cat "$SUBMISSION_RECEIPT"
