#!/bin/bash
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
RUNTIME=${RUNTIME:?RUNTIME is required}
RUNTIME_MANIFEST_SHA256=${RUNTIME_MANIFEST_SHA256:?runtime manifest hash is required}
MODEL_ROOT=${MODEL_ROOT:?MODEL_ROOT is required}
MODEL_REVISION=${MODEL_REVISION:?MODEL_REVISION is required}
MODEL_CONFIG_SHA256=${MODEL_CONFIG_SHA256:?MODEL_CONFIG_SHA256 is required}
MODEL_MANIFEST_SHA256=${MODEL_MANIFEST_SHA256:?MODEL_MANIFEST_SHA256 is required}
MATH_BANK=${MATH_BANK:?MATH_BANK is required}
MATH_BANK_SHA256=${MATH_BANK_SHA256:?MATH_BANK_SHA256 is required}
SCIENCE_BANK=${SCIENCE_BANK:?SCIENCE_BANK is required}
SCIENCE_BANK_SHA256=${SCIENCE_BANK_SHA256:?SCIENCE_BANK_SHA256 is required}
CODE_BANK=${CODE_BANK:?CODE_BANK is required}
CODE_BANK_SHA256=${CODE_BANK_SHA256:?CODE_BANK_SHA256 is required}
OUTPUT_ROOT=${OUTPUT_ROOT:?OUTPUT_ROOT is required}
SHARD_SIZE=${SHARD_SIZE:-512}
DRY_RUN=${DRY_RUN:-true}
EXCLUDE=${EXCLUDE:-evc33,evc38,evc46}

test "$SHARD_SIZE" -eq 512
test "$DRY_RUN" = true || test "$DRY_RUN" = false
test -f "$RUNTIME/train/jobs/hf_product_reasoning_rollouts.sbatch"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = \
  "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = \
  "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$MODEL_ROOT/SHA256SUMS" | cut -d' ' -f1)" = \
  "$MODEL_MANIFEST_SHA256"
(cd "$MODEL_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
for expected in \
  "$MATH_BANK_SHA256 $MATH_BANK" \
  "$SCIENCE_BANK_SHA256 $SCIENCE_BANK" \
  "$CODE_BANK_SHA256 $CODE_BANK"; do
  printf '%s\n' "$expected" | sha256sum -c - >/dev/null
done
test ! -e "$OUTPUT_ROOT"

echo "[sctr1-drafts] jobs=17 h100_per_job=1 expected_gpu_hours=8-24"
if [[ "$DRY_RUN" == false ]]; then
  mkdir -p "$OUTPUT_ROOT"
fi

submit_shard() {
  local bank_name=$1
  local bank=$2
  local skip=$3
  local shard=$4
  local count=$5
  local prefix=$OUTPUT_ROOT/draft_${bank_name}_s${shard}
  local exports
  exports="ALL,RUNTIME=$RUNTIME,MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION"
  exports+=",DATA=$bank,CANDIDATES_OUTPUT=${prefix}.candidates.jsonl"
  exports+=",POSITIVES_OUTPUT=${prefix}.positives.jsonl,REPORT=${prefix}.report.json"
  exports+=",SKIP=$skip,COUNT=$count,SAMPLES=1,GENERATION_MODE=greedy"
  exports+=",PROMPT_BATCH_SIZE=4,SEED=2026080818,MAX_NEW_TOKENS=768"
  exports+=",STAGE_MODEL=true,FINALIZE_EXHAUSTED=false,ENABLE_THINKING=false"
  exports+=",BARE_PROMPT_STYLE=reasoning"
  if [[ "$DRY_RUN" == true ]]; then
    printf 'sbatch --parsable --exclude=%q --export=%q %q\n' \
      "$EXCLUDE" "$exports" "$RUNTIME/train/jobs/hf_product_reasoning_rollouts.sbatch"
  else
    sbatch --parsable --exclude="$EXCLUDE" --export="$exports" \
      "$RUNTIME/train/jobs/hf_product_reasoning_rollouts.sbatch"
  fi
}

for bank_name in math science; do
  bank=$MATH_BANK
  if [[ "$bank_name" == science ]]; then
    bank=$SCIENCE_BANK
  fi
  for shard in $(seq 0 7); do
    submit_shard "$bank_name" "$bank" "$((shard * SHARD_SIZE))" "$shard" "$SHARD_SIZE"
  done
done
submit_shard code "$CODE_BANK" 0 0 200
