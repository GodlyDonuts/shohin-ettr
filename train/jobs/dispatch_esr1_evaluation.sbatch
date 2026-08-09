#!/bin/bash
#SBATCH --job-name=esr1-eval-dispatch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/esr1_eval_dispatch_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/esr1_eval_dispatch_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION \
  MODEL_CONFIG_SHA256 STANDARD_ROOT ALWAYS_REPORT OUTPUT_ROOT RELEASE_RECEIPT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test ! -e "$RELEASE_RECEIPT"; test -f "$ALWAYS_REPORT"

DEV=$STANDARD_ROOT/development.jsonl
DATA_REPORT=$STANDARD_ROOT/report.json
DEV_SHA=$(sha256sum "$DEV" | cut -d' ' -f1)
DATA_REPORT_SHA=$(sha256sum "$DATA_REPORT" | cut -d' ' -f1)
SYNDROME_CHECKPOINT=$OUTPUT_ROOT/syndrome_fit/checkpoint_0000256.pt
ETTR_CHECKPOINT=$OUTPUT_ROOT/workspace_control_fit/checkpoint_0000256.pt
for path in "$SYNDROME_CHECKPOINT" "$ETTR_CHECKPOINT" \
  "$OUTPUT_ROOT/syndrome_fit/report.json" \
  "$OUTPUT_ROOT/workspace_control_fit/report.json"; do test -f "$path"; done
SYNDROME_SHA=$(sha256sum "$SYNDROME_CHECKPOINT" | cut -d' ' -f1)
ETTR_SHA=$(sha256sum "$ETTR_CHECKPOINT" | cut -d' ' -f1)
EXCLUDE=${EXCLUDE:-evc33,evc38,evc46}
SHARDS=8
PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/envs/product-reasoning-b3a3603-r2/bin/python}

submit() { local raw; raw=$(sbatch --parsable "$@"); printf '%s\n' "${raw%%;*}"; }
common="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
common+=",MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION"
common+=",MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256,DATA=$DEV,DATA_SHA256=$DEV_SHA"
common+=",DATA_REPORT=$DATA_REPORT,DATA_REPORT_SHA256=$DATA_REPORT_SHA"

submit_arm() {
  local name=$1 checkpoint=$2 checkpoint_sha=$3
  local root=$OUTPUT_ROOT/development/$name
  mkdir -p "$root/shards"
  local jobs=() reports=()
  for index in $(seq 0 7); do
    local suffix candidates report job
    suffix=$(printf '%02d' "$index")
    candidates=$root/shards/candidates_$suffix.jsonl
    report=$root/shards/report_$suffix.json
    job=$(submit --exclude="$EXCLUDE" \
      --export="$common,ADAPTER_CHECKPOINT=$checkpoint,ADAPTER_CHECKPOINT_SHA256=$checkpoint_sha,CANDIDATES_OUTPUT=$candidates,REPORT=$report,SHARD_INDEX=$index,SHARD_COUNT=$SHARDS" \
      "$RUNTIME/train/jobs/hf_esr1_evaluate.sbatch")
    jobs+=("$job"); reports+=("$report")
  done
  local dependency reports_csv merge
  dependency=$(IFS=:; echo "${jobs[*]}")
  reports_csv=$(IFS=:; echo "${reports[*]}")
  merge=$(submit --dependency=afterok:"$dependency" \
    --export="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,EVALUATOR=idr1,DATA=$DEV,DATA_SHA256=$DEV_SHA,SPLIT=development,SHARD_REPORTS=$reports_csv,CANDIDATES_OUTPUT=$root/candidates.jsonl,REPORT=$root/report.json" \
    "$RUNTIME/pipeline/jobs/merge_temporal_evaluation_shards.sbatch")
  LAST_MERGE=$merge
  ALL_EVAL_JOBS+=("${jobs[@]}")
}

ALL_EVAL_JOBS=()
submit_arm syndrome "$SYNDROME_CHECKPOINT" "$SYNDROME_SHA"; syndrome_merge=$LAST_MERGE
submit_arm workspace_control "$ETTR_CHECKPOINT" "$ETTR_SHA"; ettr_merge=$LAST_MERGE
comparison=$(submit --dependency=afterok:"$syndrome_merge:$ettr_merge" \
  --export="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,SYNDROME_REPORT=$OUTPUT_ROOT/development/syndrome/report.json,ETTR_REPORT=$OUTPUT_ROOT/development/workspace_control/report.json,ALWAYS_REPORT=$ALWAYS_REPORT,SYNDROME_FIT_REPORT=$OUTPUT_ROOT/syndrome_fit/report.json,ETTR_FIT_REPORT=$OUTPUT_ROOT/workspace_control_fit/report.json,OUTPUT=$OUTPUT_ROOT/development/comparison.json" \
  "$RUNTIME/pipeline/jobs/compare_esr1_development.sbatch")

temporary=$RELEASE_RECEIPT.partial.$$
eval_csv=$(IFS=,; echo "${ALL_EVAL_JOBS[*]}")
"$PYTHON" - "$temporary" <<PY
import json, os
payload = {
  "checkpoint_sha256": {"syndrome": "$SYNDROME_SHA", "workspace_control": "$ETTR_SHA"},
  "comparison_job": "$comparison",
  "evaluation_jobs": "$eval_csv".split(","),
  "merge_jobs": ["$syndrome_merge", "$ettr_merge"],
  "status": "submitted",
}
with open(os.path.abspath("$temporary"), "x", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True); handle.write("\n")
PY
mv "$temporary" "$RELEASE_RECEIPT"; chmod a-w "$RELEASE_RECEIPT"
cat "$RELEASE_RECEIPT"
