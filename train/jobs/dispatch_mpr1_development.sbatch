#!/bin/bash
#SBATCH --job-name=mpr1-dispatch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/mpr1_dispatch_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/mpr1_dispatch_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION \
  MODEL_CONFIG_SHA256 MODEL_MANIFEST_SHA256 TRAIN_ROOT EVAL_DATA EVAL_DATA_REPORT \
  UNCHANGED_ROOT OUTPUT_ROOT RECEIPT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
EXCLUDE=${EXCLUDE:-evc29,evc33,evc38,evc46}
SHARDS=8
test ! -e "$OUTPUT_ROOT" && test ! -e "$RECEIPT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
mkdir -p "$OUTPUT_ROOT" "$(dirname "$RECEIPT")"
DATA_REPORT=$TRAIN_ROOT/report.json
DATA_REPORT_SHA=$(sha256sum "$DATA_REPORT" | cut -d' ' -f1)
ALIGNED=$TRAIN_ROOT/train_aligned.jsonl
SHUFFLED=$TRAIN_ROOT/train_shuffled.jsonl
ALIGNED_SHA=$(sha256sum "$ALIGNED" | cut -d' ' -f1)
SHUFFLED_SHA=$(sha256sum "$SHUFFLED" | cut -d' ' -f1)
EVAL_SHA=$(sha256sum "$EVAL_DATA" | cut -d' ' -f1)
EVAL_REPORT_SHA=$(sha256sum "$EVAL_DATA_REPORT" | cut -d' ' -f1)

submit() {
  local raw job
  raw=$(sbatch --parsable "$@")
  job=${raw%%;*}
  [[ $job =~ ^[0-9]+$ ]] || { echo "invalid job id: $raw" >&2; exit 5; }
  printf '%s\n' "$job"
}
common="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
common+=",MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION,MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256"
common+=",MODEL_MANIFEST_SHA256=$MODEL_MANIFEST_SHA256,DATA_REPORT=$DATA_REPORT,DATA_REPORT_SHA256=$DATA_REPORT_SHA"
mechanics=$(submit --exclude="$EXCLUDE" \
  --export="$common,DATA=$ALIGNED,DATA_SHA256=$ALIGNED_SHA,OUTPUT=$OUTPUT_ROOT/mechanics_u1,ARM=aligned,DRAFT_CONTROL=normal,UPDATES=1,CHECKPOINT_INTERVAL=1" \
  "$RUNTIME/train/jobs/train_mpr1_shared_residual.sbatch")
aligned_fit=$(submit --exclude="$EXCLUDE" --dependency=afterok:"$mechanics" \
  --export="$common,DATA=$ALIGNED,DATA_SHA256=$ALIGNED_SHA,OUTPUT=$OUTPUT_ROOT/aligned_fit,ARM=aligned,DRAFT_CONTROL=normal" \
  "$RUNTIME/train/jobs/train_mpr1_shared_residual.sbatch")
shuffled_fit=$(submit --exclude="$EXCLUDE" --dependency=afterok:"$mechanics" \
  --export="$common,DATA=$SHUFFLED,DATA_SHA256=$SHUFFLED_SHA,OUTPUT=$OUTPUT_ROOT/shuffled_fit,ARM=shuffled,DRAFT_CONTROL=normal" \
  "$RUNTIME/train/jobs/train_mpr1_shared_residual.sbatch")
hidden_fit=$(submit --exclude="$EXCLUDE" --dependency=afterok:"$mechanics" \
  --export="$common,DATA=$ALIGNED,DATA_SHA256=$ALIGNED_SHA,OUTPUT=$OUTPUT_ROOT/hidden_fit,ARM=hidden,DRAFT_CONTROL=zero_attention" \
  "$RUNTIME/train/jobs/train_mpr1_shared_residual.sbatch")

eval_common="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
eval_common+=",MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION,MODEL_LOADER=causal"
eval_common+=",MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256,MODEL_MANIFEST_SHA256=$MODEL_MANIFEST_SHA256"
eval_common+=",DATA=$EVAL_DATA,DATA_SHA256=$EVAL_SHA,DATA_REPORT=$EVAL_DATA_REPORT,DATA_REPORT_SHA256=$EVAL_REPORT_SHA,SPLIT=development,STAGE_MODEL=false"
declare -A FIT_JOBS=( [aligned]="$aligned_fit" [shuffled]="$shuffled_fit" [hidden]="$hidden_fit" )
merges=(); shard_jobs=()
for arm in aligned shuffled hidden; do
  fit=$OUTPUT_ROOT/${arm}_fit
  checkpoint=$fit/checkpoint_0000256.pt
  root=$OUTPUT_ROOT/development/$arm
  mkdir -p "$root/shards"
  jobs=(); reports=()
  for index in $(seq 0 7); do
    suffix=$(printf '%02d' "$index")
    report=$root/shards/report_$suffix.json
    job=$(submit --exclude="$EXCLUDE" --dependency=afterok:"${FIT_JOBS[$arm]}" \
      --export="$eval_common,ADAPTER_CHECKPOINT=$checkpoint,SHARD_INDEX=$index,SHARD_COUNT=$SHARDS,CANDIDATES_OUTPUT=$root/shards/candidates_$suffix.jsonl,REPORT=$report" \
      "$RUNTIME/train/jobs/hf_mpr1_evaluate_reviser.sbatch")
    jobs+=("$job"); shard_jobs+=("$job"); reports+=("$report")
  done
  dependency=$(IFS=:; echo "${jobs[*]}")
  report_csv=$(IFS=:; echo "${reports[*]}")
  merge=$(submit --dependency=afterok:"$dependency" \
    --export="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,EVALUATOR=idr1,DATA=$EVAL_DATA,DATA_SHA256=$EVAL_SHA,SPLIT=development,SHARD_REPORTS=$report_csv,CANDIDATES_OUTPUT=$root/candidates.jsonl,REPORT=$root/report.json" \
    "$RUNTIME/pipeline/jobs/merge_temporal_evaluation_shards.sbatch")
  merges+=("$merge")
done
unchanged_candidates=$UNCHANGED_ROOT/candidates.jsonl
unchanged_report=$UNCHANGED_ROOT/report.json
test -f "$unchanged_candidates" && test -f "$unchanged_report"
semantic=$OUTPUT_ROOT/development/semantic_attribution.json
semantic_job=$(submit --dependency=afterok:"${merges[0]}" \
  --export="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,UNCHANGED=$unchanged_candidates,TREATMENT=$OUTPUT_ROOT/development/aligned/candidates.jsonl,DEVELOPMENT_DATA=$EVAL_DATA,OUTPUT=$semantic" \
  "$RUNTIME/pipeline/jobs/analyze_ecr1_semantic_repairs.sbatch")
merge_dependency=$(IFS=:; echo "${merges[*]}")
comparison=$OUTPUT_ROOT/development/comparison.json
compare_job=$(submit --dependency=afterok:"$merge_dependency":"$semantic_job" \
  --export="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,ALIGNED_REPORT=$OUTPUT_ROOT/development/aligned/report.json,SHUFFLED_REPORT=$OUTPUT_ROOT/development/shuffled/report.json,HIDDEN_REPORT=$OUTPUT_ROOT/development/hidden/report.json,UNCHANGED_REPORT=$unchanged_report,ALIGNED_FIT=$OUTPUT_ROOT/aligned_fit/report.json,SHUFFLED_FIT=$OUTPUT_ROOT/shuffled_fit/report.json,HIDDEN_FIT=$OUTPUT_ROOT/hidden_fit/report.json,DATA_REPORT=$DATA_REPORT,SEMANTIC_ATTRIBUTION=$semantic,OUTPUT=$comparison" \
  "$RUNTIME/pipeline/jobs/compare_mpr1_development.sbatch")
python - "$RECEIPT" "$mechanics" "$aligned_fit" "$shuffled_fit" "$hidden_fit" \
  "$semantic_job" "$compare_job" "${shard_jobs[*]}" "${merges[*]}" <<'PY'
import json,os,sys
path,*values=sys.argv[1:]
payload={"status":"submitted","mechanics_job":values[0],"aligned_fit_job":values[1],"shuffled_fit_job":values[2],"hidden_fit_job":values[3],"semantic_job":values[4],"comparison_job":values[5],"shard_jobs":values[6].split(),"merge_jobs":values[7].split()}
tmp=f"{path}.partial.{os.getpid()}"
with open(tmp,"x") as f: json.dump(payload,f,sort_keys=True); f.write("\n")
os.replace(tmp,path)
PY
chmod a-w "$RECEIPT"
cat "$RECEIPT"
