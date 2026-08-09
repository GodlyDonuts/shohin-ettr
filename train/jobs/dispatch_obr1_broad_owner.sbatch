#!/bin/bash
#SBATCH --job-name=obr1-dispatch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/obr1_dispatch_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/obr1_dispatch_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION MODEL_CONFIG_SHA256 MODEL_MANIFEST_SHA256 TRAIN_ROOT DEVELOPMENT DEVELOPMENT_REPORT OUTPUT_ROOT RECEIPT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
EXCLUDE=${EXCLUDE:-evc29,evc33,evc38,evc46}
SHARDS=8
test ! -e "$OUTPUT_ROOT" && test ! -e "$RECEIPT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
mkdir -p "$OUTPUT_ROOT" "$(dirname "$RECEIPT")"
DATA=$TRAIN_ROOT/train.jsonl
DATA_REPORT=$TRAIN_ROOT/report.json
DATA_SHA=$(sha256sum "$DATA" | cut -d' ' -f1)
DATA_REPORT_SHA=$(sha256sum "$DATA_REPORT" | cut -d' ' -f1)
DEVELOPMENT_SHA=$(sha256sum "$DEVELOPMENT" | cut -d' ' -f1)
DEVELOPMENT_REPORT_SHA=$(sha256sum "$DEVELOPMENT_REPORT" | cut -d' ' -f1)

submit() {
  local raw job
  raw=$(sbatch --parsable "$@")
  job=${raw%%;*}
  [[ $job =~ ^[0-9]+$ ]] || { echo "invalid job id: $raw" >&2; exit 5; }
  printf '%s\n' "$job"
}
common="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION,MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256,DATA=$DATA,DATA_SHA256=$DATA_SHA,DATA_REPORT=$DATA_REPORT,DATA_REPORT_SHA256=$DATA_REPORT_SHA"
mechanics=$(submit --exclude="$EXCLUDE" --export="$common,OUTPUT=$OUTPUT_ROOT/mechanics_u1,UPDATES=1,CHECKPOINT_INTERVAL=1" "$RUNTIME/train/jobs/train_obr1_broad_owner.sbatch")
fit=$(submit --exclude="$EXCLUDE" --dependency=afterok:"$mechanics" --export="$common,OUTPUT=$OUTPUT_ROOT/fit" "$RUNTIME/train/jobs/train_obr1_broad_owner.sbatch")
checkpoint=$OUTPUT_ROOT/fit/checkpoint_0002048.pt
jobs=(); reports=()
for index in $(seq 0 7); do
  suffix=$(printf '%02d' "$index")
  report=$OUTPUT_ROOT/development/shards/report_$suffix.json
  mkdir -p "$OUTPUT_ROOT/development/shards"
  job=$(submit --exclude="$EXCLUDE" --dependency=afterok:"$fit" \
    --export="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION,MODEL_LOADER=causal,MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256,MODEL_MANIFEST_SHA256=$MODEL_MANIFEST_SHA256,DATA=$DEVELOPMENT,DATA_SHA256=$DEVELOPMENT_SHA,DATA_REPORT=$DEVELOPMENT_REPORT,DATA_REPORT_SHA256=$DEVELOPMENT_REPORT_SHA,SPLIT=development,STAGE_MODEL=false,ADAPTER_CHECKPOINT=$checkpoint,SHARD_INDEX=$index,SHARD_COUNT=$SHARDS,CANDIDATES_OUTPUT=$OUTPUT_ROOT/development/shards/candidates_$suffix.jsonl,REPORT=$report" \
    "$RUNTIME/train/jobs/hf_mpr1_evaluate_reviser.sbatch")
  jobs+=("$job"); reports+=("$report")
done
dependency=$(IFS=:; echo "${jobs[*]}")
report_csv=$(IFS=:; echo "${reports[*]}")
merge=$(submit --dependency=afterok:"$dependency" \
  --export="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,EVALUATOR=idr1,DATA=$DEVELOPMENT,DATA_SHA256=$DEVELOPMENT_SHA,SPLIT=development,SHARD_REPORTS=$report_csv,CANDIDATES_OUTPUT=$OUTPUT_ROOT/development/candidates.jsonl,REPORT=$OUTPUT_ROOT/development/report.json" \
  "$RUNTIME/pipeline/jobs/merge_temporal_evaluation_shards.sbatch")
compare=$(submit --dependency=afterok:"$merge" \
  --export="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,EVALUATION_REPORT=$OUTPUT_ROOT/development/report.json,FIT_REPORT=$OUTPUT_ROOT/fit/report.json,DATA_REPORT=$DATA_REPORT,OUTPUT=$OUTPUT_ROOT/development/comparison.json" \
  "$RUNTIME/pipeline/jobs/compare_obr1_development.sbatch")
python3 - "$RECEIPT" "$mechanics" "$fit" "$merge" "$compare" "${jobs[*]}" <<'PY'
import json,os,sys
path,*v=sys.argv[1:]
payload={"status":"submitted","mechanics_job":v[0],"fit_job":v[1],"merge_job":v[2],"comparison_job":v[3],"evaluation_jobs":v[4].split()}
tmp=f"{path}.partial.{os.getpid()}"
with open(tmp,"x") as f: json.dump(payload,f,sort_keys=True); f.write("\n")
os.replace(tmp,path)
PY
chmod a-w "$RECEIPT"
cat "$RECEIPT"
