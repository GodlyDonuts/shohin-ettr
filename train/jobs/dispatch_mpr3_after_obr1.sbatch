#!/bin/bash
#SBATCH --job-name=mpr3-release
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/mpr3_release_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/mpr3_release_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION MODEL_CONFIG_SHA256 MODEL_MANIFEST_SHA256 OBR_ROOT MPR1_DATA_ROOT DEVELOPMENT DEVELOPMENT_REPORT DATA_ROOT OUTPUT_ROOT RECEIPT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
EXCLUDE=${EXCLUDE:-evc29,evc33,evc38,evc46}
SHARDS=8
test ! -e "$DATA_ROOT" && test ! -e "$OUTPUT_ROOT" && test ! -e "$RECEIPT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
OWNER_CHECKPOINT=$OBR_ROOT/fit/checkpoint_0002048.pt
OWNER_CANDIDATES=$OBR_ROOT/development/candidates.jsonl
OWNER_REPORT=$OBR_ROOT/development/report.json
OBR_COMPARISON=$OBR_ROOT/development/comparison.json
for path in "$OWNER_CHECKPOINT" "$OWNER_CANDIDATES" "$OWNER_REPORT" "$OBR_COMPARISON"; do test -f "$path"; done
python3 - "$OBR_COMPARISON" "$OWNER_REPORT" <<'PY'
import json,sys
comparison,owner=map(lambda p:json.load(open(p)),sys.argv[1:])
assert comparison.get("schema")=="shohin-obr1-development-comparison-v1"
assert comparison.get("status")=="pass" and comparison.get("qualified_for_temporal_gate") is True
assert int(comparison.get("score",-1))>=300
assert owner.get("schema")=="shohin-idr1-revision-evaluation-v1" and owner.get("status")=="complete"
assert int(owner["metrics"]["overall"]["generated_correct"])==int(comparison["score"])
PY
mkdir -p "$OUTPUT_ROOT/train_drafts/shards" "$(dirname "$RECEIPT")"
SOURCE=$MPR1_DATA_ROOT/train_aligned.jsonl
SOURCE_REPORT=$MPR1_DATA_ROOT/report.json
SOURCE_SHA=$(sha256sum "$SOURCE" | cut -d' ' -f1)
SOURCE_REPORT_SHA=$(sha256sum "$SOURCE_REPORT" | cut -d' ' -f1)
OWNER_SHA=$(sha256sum "$OWNER_CHECKPOINT" | cut -d' ' -f1)

submit() {
  local raw job
  raw=$(sbatch --parsable "$@")
  job=${raw%%;*}
  [[ $job =~ ^[0-9]+$ ]] || { echo "invalid job id: $raw" >&2; exit 5; }
  printf '%s\n' "$job"
}
draft_jobs=(); draft_reports=()
for index in $(seq 0 7); do
  suffix=$(printf '%02d' "$index")
  report=$OUTPUT_ROOT/train_drafts/shards/report_$suffix.json
  job=$(submit --exclude="$EXCLUDE" \
    --export="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION,MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256,SOURCE=$SOURCE,SOURCE_SHA256=$SOURCE_SHA,SOURCE_REPORT=$SOURCE_REPORT,SOURCE_REPORT_SHA256=$SOURCE_REPORT_SHA,OWNER_CHECKPOINT=$OWNER_CHECKPOINT,OWNER_CHECKPOINT_SHA256=$OWNER_SHA,EXPECTED_OWNER_UPDATE=2048,OUTPUT=$OUTPUT_ROOT/train_drafts/shards/drafts_$suffix.jsonl,REPORT=$report,SHARD_INDEX=$index,SHARD_COUNT=$SHARDS" \
    "$RUNTIME/train/jobs/hf_mpr2_generate_drafts.sbatch")
  draft_jobs+=("$job"); draft_reports+=("$report")
done
draft_dependency=$(IFS=:; echo "${draft_jobs[*]}")
draft_report_csv=$(IFS=:; echo "${draft_reports[*]}")
merge=$(submit --dependency=afterok:"$draft_dependency" \
  --export="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,SHARD_REPORTS=$draft_report_csv,OUTPUT=$OUTPUT_ROOT/train_drafts/drafts.jsonl,REPORT=$OUTPUT_ROOT/train_drafts/report.json" \
  "$RUNTIME/pipeline/jobs/merge_mpr2_draft_shards.sbatch")
data=$(submit --dependency=afterok:"$merge" \
  --export="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,MODEL_ROOT=$MODEL_ROOT,TRAIN=$SOURCE,TRAIN_REPORT=$SOURCE_REPORT,TRAIN_DRAFTS=$OUTPUT_ROOT/train_drafts/drafts.jsonl,TRAIN_DRAFT_REPORT=$OUTPUT_ROOT/train_drafts/report.json,DEVELOPMENT=$DEVELOPMENT,DEVELOPMENT_REPORT=$DEVELOPMENT_REPORT,DEVELOPMENT_OWNER_CANDIDATES=$OWNER_CANDIDATES,DEVELOPMENT_OWNER_REPORT=$OWNER_REPORT,OUTPUT=$DATA_ROOT" \
  "$RUNTIME/pipeline/jobs/build_mpr2_revision_data.sbatch")
campaign=$(submit --dependency=afterok:"$data" \
  --export="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION,MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256,MODEL_MANIFEST_SHA256=$MODEL_MANIFEST_SHA256,DATA_ROOT=$DATA_ROOT,OUTPUT_ROOT=$OUTPUT_ROOT/campaign,OWNER_CANDIDATES=$OWNER_CANDIDATES,OWNER_REPORT=$OWNER_REPORT,RECEIPT=$OUTPUT_ROOT/campaign_dispatch_receipt.json,COMPARISON=mpr3" \
  "$RUNTIME/train/jobs/dispatch_mpr2_after_data.sbatch")
python3 - "$RECEIPT" "$merge" "$data" "$campaign" "${draft_jobs[*]}" <<'PY'
import json,os,sys
path,*v=sys.argv[1:]
payload={"status":"submitted","draft_merge_job":v[0],"data_job":v[1],"campaign_dispatch_job":v[2],"draft_jobs":v[3].split()}
tmp=f"{path}.partial.{os.getpid()}"
with open(tmp,"x") as f: json.dump(payload,f,sort_keys=True); f.write("\n")
os.replace(tmp,path)
PY
chmod a-w "$RECEIPT"
cat "$RECEIPT"
