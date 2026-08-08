#!/bin/bash
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
RUNTIME=${RUNTIME:-$BASE/runtime/cvg1_rollouts_dd4ef87_r1}
RUNTIME_MANIFEST_SHA256=${RUNTIME_MANIFEST_SHA256:-ff44c04391871a4af206fcf9fe842e84b56495da30b87d81d5cb5da850943229}
MODEL_ROOT=${MODEL_ROOT:-$BASE/artifacts/external/qwen3.5-4b-851bf6e}
MODEL_REVISION=${MODEL_REVISION:-851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}
MODEL_CONFIG_SHA256=${MODEL_CONFIG_SHA256:-ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670}
BASE_CHECKPOINT=${BASE_CHECKPOINT:-$BASE/artifacts/product_reasoning/qwen3.5-4b-851bf6e/qpt1_b1_u256_402fa8e/checkpoint_0000256.pt}
BASE_CHECKPOINT_SHA256=${BASE_CHECKPOINT_SHA256:-f7354e6a0c4311ad792b73358b4e62d9dbe0ae1bd2d41896cf55482d9ce81feb}
EXPERT_CHECKPOINT=${EXPERT_CHECKPOINT:-$BASE/artifacts/product_reasoning/qwen3.5-4b-851bf6e/qpt1_u256_402fa8e/checkpoint_0000256.pt}
EXPERT_CHECKPOINT_SHA256=${EXPERT_CHECKPOINT_SHA256:-97351d9b572b371ff09a7f675ef4b1893c20b69739727b4a207ec5a3c9813350}
BANK_ROOT=${BANK_ROOT:-$BASE/artifacts/product_reasoning/router_outcomes/cvg1_disjoint_5e63a06}
MATH_BANK=${MATH_BANK:-$BANK_ROOT/math_disjoint4096_2026080709_r2.jsonl}
MATH_BANK_SHA256=${MATH_BANK_SHA256:-e0ede83257e441050a019f59fb13d9c85bd6cba1d6a755ab86fb7129966ddbe5}
SCIENCE_BANK=${SCIENCE_BANK:-$BANK_ROOT/science_disjoint4096_2026080709_r2.jsonl}
SCIENCE_BANK_SHA256=${SCIENCE_BANK_SHA256:-5a96859fd9088cde598b61da60dd2c6cb7281323ee06c034742a1b4e0e237017}
OUTPUT_ROOT=${OUTPUT_ROOT:-$BASE/artifacts/product_reasoning/cvg1_rollouts_dd4ef87_r1}
SHARD_SIZE=${SHARD_SIZE:-512}
DRY_RUN=${DRY_RUN:-true}

test "$SHARD_SIZE" -eq 512
test "$DRY_RUN" = true || test "$DRY_RUN" = false
test -f "$RUNTIME/train/jobs/hf_product_reasoning_rollouts.sbatch"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$BASE_CHECKPOINT" | cut -d' ' -f1)" = "$BASE_CHECKPOINT_SHA256"
test "$(sha256sum "$EXPERT_CHECKPOINT" | cut -d' ' -f1)" = "$EXPERT_CHECKPOINT_SHA256"
test "$(sha256sum "$MATH_BANK" | cut -d' ' -f1)" = "$MATH_BANK_SHA256"
test "$(sha256sum "$SCIENCE_BANK" | cut -d' ' -f1)" = "$SCIENCE_BANK_SHA256"
test ! -e "$OUTPUT_ROOT"

echo "[cvg1-rollouts] jobs=32 h100_per_job=1 expected_gpu_hours=26-32"
if [[ "$DRY_RUN" == false ]]; then
  mkdir -p "$OUTPUT_ROOT"
fi

submit_shard() {
  local lineage=$1
  local checkpoint=$2
  local bank_name=$3
  local bank=$4
  local skip=$5
  local shard=$6
  local prefix=$OUTPUT_ROOT/${lineage}_${bank_name}_s${shard}
  local exports
  exports="ALL,RUNTIME=$RUNTIME,MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION"
  exports+=",ADAPTER_CHECKPOINT=$checkpoint,DATA=$bank"
  exports+=",CANDIDATES_OUTPUT=${prefix}.candidates.jsonl"
  exports+=",POSITIVES_OUTPUT=${prefix}.positives.jsonl,REPORT=${prefix}.report.json"
  exports+=",SKIP=$skip,COUNT=$SHARD_SIZE,SAMPLES=1,GENERATION_MODE=greedy"
  exports+=",PROMPT_BATCH_SIZE=4,SEED=2026080714,MAX_NEW_TOKENS=768"
  exports+=",STAGE_MODEL=true,FINALIZE_EXHAUSTED=false,ENABLE_THINKING=false"
  exports+=",BARE_PROMPT_STYLE=reasoning"
  if [[ "$DRY_RUN" == true ]]; then
    printf 'sbatch --parsable --export=%q %q\n' \
      "$exports" "$RUNTIME/train/jobs/hf_product_reasoning_rollouts.sbatch"
  else
    sbatch --parsable --export="$exports" \
      "$RUNTIME/train/jobs/hf_product_reasoning_rollouts.sbatch"
  fi
}

for lineage in base expert; do
  checkpoint=$BASE_CHECKPOINT
  if [[ "$lineage" == expert ]]; then
    checkpoint=$EXPERT_CHECKPOINT
  fi
  for bank_name in math science; do
    bank=$MATH_BANK
    if [[ "$bank_name" == science ]]; then
      bank=$SCIENCE_BANK
    fi
    for shard in $(seq 0 7); do
      skip=$((shard * SHARD_SIZE))
      submit_shard "$lineage" "$checkpoint" "$bank_name" "$bank" "$skip" "$shard"
    done
  done
done
