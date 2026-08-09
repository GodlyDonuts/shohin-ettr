#!/bin/bash
#SBATCH --job-name=rcr1-dev-dispatch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/rcr1_dispatch_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/rcr1_dispatch_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION \
  MODEL_CONFIG_SHA256 MODEL_MANIFEST_SHA256 DATA_ROOT ROUTER_FIT ATTENTION_FIT \
  MTR_ROOT OUTPUT_ROOT DISPATCH_RECEIPT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
SHARDS=8
EXCLUDE=${EXCLUDE:-evc33,evc38,evc46}
test ! -e "$OUTPUT_ROOT"; test ! -e "$DISPATCH_RECEIPT"
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
common+=",DATA_REPORT_SHA256=$DATA_REPORT_SHA256,SPLIT=development"
merges=(); shards=()
for arm in router attention; do
  fit_var=${arm^^}_FIT; fit=${!fit_var}
  checkpoint=$fit/checkpoint_0000256.pt; test -f "$checkpoint"
  checkpoint_sha=$(sha256sum "$checkpoint" | cut -d' ' -f1)
  root=$OUTPUT_ROOT/$arm; mkdir -p "$root/shards"
  jobs=(); reports=()
  for index in $(seq 0 7); do
    suffix=$(printf '%02d' "$index")
    report=$root/shards/report_$suffix.json
    job=$(submit --exclude="$EXCLUDE" \
      --export="$common,SHARD_INDEX=$index,SHARD_COUNT=$SHARDS,ADAPTER_CHECKPOINT=$checkpoint,ADAPTER_CHECKPOINT_SHA256=$checkpoint_sha,CANDIDATES_OUTPUT=$root/shards/candidates_$suffix.jsonl,REPORT=$report" \
      "$RUNTIME/train/jobs/hf_idr1_evaluate_reviser.sbatch")
    jobs+=("$job"); shards+=("$job"); reports+=("$report")
  done
  dependency=$(IFS=:; echo "${jobs[*]}")
  report_csv=$(IFS=:; echo "${reports[*]}")
  merge=$(submit --dependency=afterok:"$dependency" \
    --export="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,EVALUATOR=idr1,DATA=$DATA,DATA_SHA256=$DATA_SHA256,SPLIT=development,SHARD_REPORTS=$report_csv,CANDIDATES_OUTPUT=$root/candidates.jsonl,REPORT=$root/report.json" \
    "$RUNTIME/pipeline/jobs/merge_temporal_evaluation_shards.sbatch")
  merges+=("$merge")
done
dependency=$(IFS=:; echo "${merges[*]}")
comparison=$(submit --dependency=afterok:"$dependency" \
  --export="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,ROUTER_REPORT=$OUTPUT_ROOT/router/report.json,ATTENTION_REPORT=$OUTPUT_ROOT/attention/report.json,UNCHANGED_REPORT=$MTR_ROOT/development/unchanged_second_pass/report.json,PRIOR_TREATMENT_REPORT=$MTR_ROOT/development/treatment/report.json,ROUTER_FIT_REPORT=$ROUTER_FIT/report.json,ATTENTION_FIT_REPORT=$ATTENTION_FIT/report.json,OUTPUT=$OUTPUT_ROOT/comparison.json" \
  "$RUNTIME/pipeline/jobs/compare_rcr1_development.sbatch")
temporary=$DISPATCH_RECEIPT.partial.$$
printf '{"comparison_job":"%s","merge_jobs":"%s","shard_jobs":"%s","status":"submitted"}\n' \
  "$comparison" "$(IFS=,; echo "${merges[*]}")" "$(IFS=,; echo "${shards[*]}")" >"$temporary"
mv "$temporary" "$DISPATCH_RECEIPT"; chmod a-w "$DISPATCH_RECEIPT"
cat "$DISPATCH_RECEIPT"
