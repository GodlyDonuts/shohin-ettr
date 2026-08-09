#!/bin/bash
#SBATCH --job-name=ecr1-dev-dispatch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/ecr1_dispatch_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/ecr1_dispatch_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION \
  MODEL_CONFIG_SHA256 MODEL_MANIFEST_SHA256 DATA_ROOT MTR_ROOT ECR_ROOT \
  OUTPUT_ROOT DISPATCH_RECEIPT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
SHARDS=8
EXCLUDE=${EXCLUDE:-evc29,evc33,evc38,evc46}
test ! -e "$OUTPUT_ROOT" && test ! -e "$DISPATCH_RECEIPT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
DATA=$DATA_ROOT/development.jsonl
DATA_REPORT=$DATA_ROOT/report.json
DATA_SHA256=$(sha256sum "$DATA" | cut -d' ' -f1)
DATA_REPORT_SHA256=$(sha256sum "$DATA_REPORT" | cut -d' ' -f1)
mkdir -p "$OUTPUT_ROOT" "$(dirname "$DISPATCH_RECEIPT")"
submit() { local raw; raw=$(sbatch --parsable "$@"); printf '%s\n' "${raw%%;*}"; }
common="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
common+=",MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION,MODEL_LOADER=causal"
common+=",MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256,MODEL_MANIFEST_SHA256=$MODEL_MANIFEST_SHA256"
common+=",DATA=$DATA,DATA_SHA256=$DATA_SHA256,DATA_REPORT=$DATA_REPORT"
common+=",DATA_REPORT_SHA256=$DATA_REPORT_SHA256,SPLIT=development,STAGE_MODEL=false"
merges=(); shards=(); treatment_merge=""
for arm in treatment shared draft_unavailable code_zero code_mean code_permutation; do
  case "$arm" in
    treatment) fit=$ECR_ROOT/treatment_fit; intervention=normal ;;
    shared) fit=$ECR_ROOT/shared_fit; intervention=normal ;;
    draft_unavailable) fit=$ECR_ROOT/draft_unavailable_fit; intervention=normal ;;
    code_zero) fit=$ECR_ROOT/treatment_fit; intervention=zero ;;
    code_mean) fit=$ECR_ROOT/treatment_fit; intervention=mean ;;
    code_permutation) fit=$ECR_ROOT/treatment_fit; intervention=permutation ;;
  esac
  checkpoint=$fit/checkpoint_0000256.pt
  test -f "$checkpoint" && test -f "$fit/report.json"
  checkpoint_sha=$(sha256sum "$checkpoint" | cut -d' ' -f1)
  root=$OUTPUT_ROOT/$arm
  mkdir -p "$root/shards"
  jobs=(); reports=()
  for index in $(seq 0 7); do
    suffix=$(printf '%02d' "$index")
    report=$root/shards/report_$suffix.json
    job=$(submit --exclude="$EXCLUDE" \
      --export="$common,SHARD_INDEX=$index,SHARD_COUNT=$SHARDS,ECR_CODE_INTERVENTION=$intervention,ADAPTER_CHECKPOINT=$checkpoint,ADAPTER_CHECKPOINT_SHA256=$checkpoint_sha,CANDIDATES_OUTPUT=$root/shards/candidates_$suffix.jsonl,REPORT=$report" \
      "$RUNTIME/train/jobs/hf_idr1_evaluate_reviser.sbatch")
    jobs+=("$job"); shards+=("$job"); reports+=("$report")
  done
  dependency=$(IFS=:; echo "${jobs[*]}")
  report_csv=$(IFS=:; echo "${reports[*]}")
  merge=$(submit --dependency=afterok:"$dependency" \
    --export="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,EVALUATOR=idr1,DATA=$DATA,DATA_SHA256=$DATA_SHA256,SPLIT=development,SHARD_REPORTS=$report_csv,CANDIDATES_OUTPUT=$root/candidates.jsonl,REPORT=$root/report.json" \
    "$RUNTIME/pipeline/jobs/merge_temporal_evaluation_shards.sbatch")
  merges+=("$merge")
  [[ "$arm" == treatment ]] && treatment_merge=$merge
done
semantic=$OUTPUT_ROOT/semantic_attribution.json
semantic_job=$(submit --dependency=afterok:"$treatment_merge" \
  --export="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,UNCHANGED=$MTR_ROOT/development/unchanged_second_pass/candidates.jsonl,TREATMENT=$OUTPUT_ROOT/treatment/candidates.jsonl,DEVELOPMENT_DATA=$DATA,OUTPUT=$semantic" \
  "$RUNTIME/pipeline/jobs/analyze_ecr1_semantic_repairs.sbatch")
dependency=$(IFS=:; echo "${merges[*]}"):$semantic_job
comparison=$(submit --dependency=afterok:"$dependency" \
  --export="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,TREATMENT_REPORT=$OUTPUT_ROOT/treatment/report.json,SHARED_REPORT=$OUTPUT_ROOT/shared/report.json,DRAFT_REPORT=$OUTPUT_ROOT/draft_unavailable/report.json,ZERO_REPORT=$OUTPUT_ROOT/code_zero/report.json,MEAN_REPORT=$OUTPUT_ROOT/code_mean/report.json,PERMUTATION_REPORT=$OUTPUT_ROOT/code_permutation/report.json,UNCHANGED_REPORT=$MTR_ROOT/development/unchanged_second_pass/report.json,MTR_REPORT=$MTR_ROOT/development/treatment/report.json,TREATMENT_FIT=$ECR_ROOT/treatment_fit/report.json,SHARED_FIT=$ECR_ROOT/shared_fit/report.json,DRAFT_FIT=$ECR_ROOT/draft_unavailable_fit/report.json,SEMANTIC_REPORT=$semantic,OUTPUT=$OUTPUT_ROOT/comparison.json" \
  "$RUNTIME/pipeline/jobs/compare_ecr1_development.sbatch")
temporary=$DISPATCH_RECEIPT.partial.$$
printf '{"comparison_job":"%s","merge_jobs":"%s","semantic_job":"%s","shard_jobs":"%s","status":"submitted"}\n' \
  "$comparison" "$(IFS=,; echo "${merges[*]}")" "$semantic_job" \
  "$(IFS=,; echo "${shards[*]}")" >"$temporary"
mv "$temporary" "$DISPATCH_RECEIPT"
chmod a-w "$DISPATCH_RECEIPT"
cat "$DISPATCH_RECEIPT"
