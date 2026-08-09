#!/bin/bash
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
RUNTIME=${RUNTIME:?set immutable runtime}
RUNTIME_MANIFEST_SHA256=${RUNTIME_MANIFEST_SHA256:?set runtime manifest hash}
MODEL_ROOT=${MODEL_ROOT:?set pinned model root}
MODEL_REVISION=${MODEL_REVISION:?set pinned model revision}
MODEL_CONFIG_SHA256=${MODEL_CONFIG_SHA256:?set model config hash}
MODEL_MANIFEST_SHA256=${MODEL_MANIFEST_SHA256:?set model manifest hash}
BANK_ROOT=${BANK_ROOT:-$BASE/artifacts/product_reasoning/router_outcomes/cvg1_disjoint_5e63a06}
MATH_BANK=${MATH_BANK:-$BANK_ROOT/math_disjoint4096_2026080709_r2.jsonl}
MATH_BANK_SHA256=${MATH_BANK_SHA256:-e0ede83257e441050a019f59fb13d9c85bd6cba1d6a755ab86fb7129966ddbe5}
SCIENCE_BANK=${SCIENCE_BANK:-$BANK_ROOT/science_disjoint4096_2026080709_r2.jsonl}
SCIENCE_BANK_SHA256=${SCIENCE_BANK_SHA256:-5a96859fd9088cde598b61da60dd2c6cb7281323ee06c034742a1b4e0e237017}
CODE_BANK_ROOT=${CODE_BANK_ROOT:-$BASE/artifacts/product_reasoning/router_outcomes/cvg1_code_disjoint_1b49ca7}
CODE_BANK_MANIFEST_SHA256=${CODE_BANK_MANIFEST_SHA256:-ce58d5104c1b9c7d6f5dbdedd3147c75cbbc0cab80f0ac3033ab539b0763e93e}
CODE_BANK=${CODE_BANK:-$CODE_BANK_ROOT/mbpp_disjoint200_2026080715_r1.jsonl}
CODE_BANK_SHA256=${CODE_BANK_SHA256:-0b6d068b4d71f407cb234579b9278dc640df09139ea906dd0f52a6ab71e05398}
OUTPUT_ROOT=${OUTPUT_ROOT:?set fresh draft output root}
SHARD_SIZE=${SHARD_SIZE:-512}
DRY_RUN=${DRY_RUN:-true}
EXCLUDE=${EXCLUDE:-evc33,evc38,evc46}

test "$SHARD_SIZE" -eq 512
test "$DRY_RUN" = true || test "$DRY_RUN" = false
test -f "$RUNTIME/train/jobs/hf_product_reasoning_rollouts.sbatch"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$MODEL_ROOT/SHA256SUMS" | cut -d' ' -f1)" = "$MODEL_MANIFEST_SHA256"
(cd "$MODEL_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MATH_BANK" | cut -d' ' -f1)" = "$MATH_BANK_SHA256"
test "$(sha256sum "$SCIENCE_BANK" | cut -d' ' -f1)" = "$SCIENCE_BANK_SHA256"
test "$(sha256sum "$CODE_BANK_ROOT/SHA256SUMS" | cut -d' ' -f1)" = "$CODE_BANK_MANIFEST_SHA256"
(cd "$CODE_BANK_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$CODE_BANK" | cut -d' ' -f1)" = "$CODE_BANK_SHA256"
test ! -e "$OUTPUT_ROOT"

echo "[transfer-drafts] jobs=17 h100_per_job=1 expected_gpu_hours=2-8"
if [[ "$DRY_RUN" == false ]]; then mkdir -p "$OUTPUT_ROOT"; fi

submit_shard() {
  local bank_name=$1 bank=$2 skip=$3 shard=$4 count=$5
  local prefix=$OUTPUT_ROOT/draft_${bank_name}_s${shard}
  local exports
  exports="ALL,RUNTIME=$RUNTIME,MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION"
  exports+=",MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256,MODEL_MANIFEST_SHA256=$MODEL_MANIFEST_SHA256"
  exports+=",DATA=$bank,CANDIDATES_OUTPUT=${prefix}.candidates.jsonl"
  exports+=",POSITIVES_OUTPUT=${prefix}.positives.jsonl,REPORT=${prefix}.report.json"
  exports+=",SKIP=$skip,COUNT=$count,SAMPLES=1,GENERATION_MODE=greedy"
  exports+=",PROMPT_BATCH_SIZE=4,SEED=2026080818,MAX_NEW_TOKENS=768"
  exports+=",STAGE_MODEL=true,FINALIZE_EXHAUSTED=false,ENABLE_THINKING=false"
  exports+=",BARE_PROMPT_STYLE=reasoning,QUANTIZATION=none"
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
  if [[ "$bank_name" == science ]]; then bank=$SCIENCE_BANK; fi
  for shard in $(seq 0 7); do
    submit_shard "$bank_name" "$bank" "$((shard * SHARD_SIZE))" "$shard" "$SHARD_SIZE"
  done
done
submit_shard code "$CODE_BANK" 0 0 200

