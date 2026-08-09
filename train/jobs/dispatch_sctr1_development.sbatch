#!/bin/bash
#SBATCH --job-name=sctr1-dev-dispatch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/sctr1_dispatch_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/sctr1_dispatch_%j.err

set -euo pipefail

for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION \
  MODEL_CONFIG_SHA256 SCTR_ROOT STANDARD_ROOT OUTPUT_ROOT DISPATCH_RECEIPT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = \
  "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = \
  "$MODEL_CONFIG_SHA256"
test ! -e "$OUTPUT_ROOT"
test ! -e "$DISPATCH_RECEIPT"

for root in "$SCTR_ROOT" "$STANDARD_ROOT"; do
  test -f "$root/train.jsonl"
  test -f "$root/development.jsonl"
  test -f "$root/report.json"
done
test -f "$SCTR_ROOT/train_shuffled.jsonl"

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test -x "$PYTHON"
"$PYTHON" - "$SCTR_ROOT" "$STANDARD_ROOT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

for root_text, schema, keys in (
    (sys.argv[1], "shohin-sctr1-selective-commit-data-report-v1", ("train", "train_shuffled", "development")),
    (sys.argv[2], "shohin-idr1-revision-data-report-v1", ("train", "development")),
):
    root = Path(root_text).resolve()
    report_path = root / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != schema or report.get("status") != "complete":
        raise SystemExit(f"temporal data report differs: {root}")
    for key in keys:
        path = root / f"{key}.jsonl"
        expected = report.get("outputs", {}).get(key, {})
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if Path(expected.get("path", "")).resolve() != path or expected.get("sha256") != digest:
            raise SystemExit(f"temporal data receipt differs: {path}")
PY

mkdir -p "$OUTPUT_ROOT" "$(dirname "$DISPATCH_RECEIPT")"
SCTR_REPORT=$SCTR_ROOT/report.json
SCTR_REPORT_SHA256=$(sha256sum "$SCTR_REPORT" | cut -d' ' -f1)
STANDARD_REPORT=$STANDARD_ROOT/report.json
STANDARD_REPORT_SHA256=$(sha256sum "$STANDARD_REPORT" | cut -d' ' -f1)
SCTR_TRAIN=$SCTR_ROOT/train.jsonl
SCTR_TRAIN_SHA256=$(sha256sum "$SCTR_TRAIN" | cut -d' ' -f1)
SCTR_SHUFFLED=$SCTR_ROOT/train_shuffled.jsonl
SCTR_SHUFFLED_SHA256=$(sha256sum "$SCTR_SHUFFLED" | cut -d' ' -f1)
SCTR_DEV=$SCTR_ROOT/development.jsonl
SCTR_DEV_SHA256=$(sha256sum "$SCTR_DEV" | cut -d' ' -f1)
STANDARD_TRAIN=$STANDARD_ROOT/train.jsonl
STANDARD_TRAIN_SHA256=$(sha256sum "$STANDARD_TRAIN" | cut -d' ' -f1)
STANDARD_DEV=$STANDARD_ROOT/development.jsonl
STANDARD_DEV_SHA256=$(sha256sum "$STANDARD_DEV" | cut -d' ' -f1)
SHARDS=${SHARDS:-8}
test "$SHARDS" -eq 8
EXCLUDE=${EXCLUDE:-evc33,evc38,evc46}

submit() {
  local raw
  raw=$(sbatch --parsable "$@")
  printf '%s\n' "${raw%%;*}"
}

common="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
common+=",MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION"
common+=",MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256,MODEL_LOADER=auto"

submit_fit() {
  local name=$1 kind=$2 data=$3 data_sha=$4 report=$5 report_sha=$6 data_key=$7
  submit --exclude="$EXCLUDE" \
    --export="$common,DATA=$data,DATA_SHA256=$data_sha,DATA_REPORT=$report,DATA_REPORT_SHA256=$report_sha,OUTPUT=$OUTPUT_ROOT/${name}_fit,TRAIN_KIND=$kind,DATA_KEY=$data_key,UPDATES=256" \
    "$RUNTIME/train/jobs/hf_sctr1_train.sbatch"
}

selective_fit=$(submit_fit selective selective_commit "$SCTR_TRAIN" "$SCTR_TRAIN_SHA256" "$SCTR_REPORT" "$SCTR_REPORT_SHA256" train)
shuffled_fit=$(submit_fit shuffled selective_commit "$SCTR_SHUFFLED" "$SCTR_SHUFFLED_SHA256" "$SCTR_REPORT" "$SCTR_REPORT_SHA256" train_shuffled)
independent_fit=$(submit_fit independent independent_commitment "$SCTR_TRAIN" "$SCTR_TRAIN_SHA256" "$SCTR_REPORT" "$SCTR_REPORT_SHA256" train)
always_fit=$(submit_fit always_revise always_revise "$STANDARD_TRAIN" "$STANDARD_TRAIN_SHA256" "$STANDARD_REPORT" "$STANDARD_REPORT_SHA256" train)

all_eval_jobs=()
merge_jobs=()

submit_trained_arm() {
  local name=$1 evaluator=$2 fit_job=$3 checkpoint=$4 data=$5 data_sha=$6 report=$7 report_sha=$8 mask=$9
  local root=$OUTPUT_ROOT/development/$name
  mkdir -p "$root/shards"
  local jobs=() reports=()
  for index in $(seq 0 $((SHARDS - 1))); do
    local suffix candidates shard_report job
    suffix=$(printf '%02d' "$index")
    candidates=$root/shards/candidates_$suffix.jsonl
    shard_report=$root/shards/report_$suffix.json
    job=$(submit --exclude="$EXCLUDE" --dependency=afterok:"$fit_job" \
      --export="$common,EVALUATOR=$evaluator,CHECKPOINT=$checkpoint,DATA=$data,DATA_SHA256=$data_sha,DATA_REPORT=$report,DATA_REPORT_SHA256=$report_sha,SPLIT=development,CANDIDATES_OUTPUT=$candidates,REPORT=$shard_report,SHARD_INDEX=$index,SHARD_COUNT=$SHARDS,MASK_INTERNAL_DRAFT=$mask" \
      "$RUNTIME/train/jobs/hf_temporal_evaluate_after_fit.sbatch")
    jobs+=("$job")
    all_eval_jobs+=("$job")
    reports+=("$shard_report")
  done
  local dependency reports_csv merge
  dependency=$(IFS=:; echo "${jobs[*]}")
  reports_csv=$(IFS=:; echo "${reports[*]}")
  merge=$(submit --dependency=afterok:"$dependency" \
    --export=ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,EVALUATOR=$evaluator,DATA=$data,DATA_SHA256=$data_sha,SPLIT=development,SHARD_REPORTS=$reports_csv,CANDIDATES_OUTPUT=$root/candidates.jsonl,REPORT=$root/report.json \
    "$RUNTIME/pipeline/jobs/merge_temporal_evaluation_shards.sbatch")
  merge_jobs+=("$merge")
  LAST_MERGE=$merge
}

submit_base_control() {
  local name=$1
  local root=$OUTPUT_ROOT/development/$name
  mkdir -p "$root/shards"
  local jobs=() reports=()
  for index in $(seq 0 $((SHARDS - 1))); do
    local suffix candidates shard_report job
    suffix=$(printf '%02d' "$index")
    candidates=$root/shards/candidates_$suffix.jsonl
    shard_report=$root/shards/report_$suffix.json
    job=$(submit --exclude="$EXCLUDE" \
      --export="$common,CONTROL=$name,DATA=$SCTR_DEV,DATA_SHA256=$SCTR_DEV_SHA256,DATA_REPORT=$SCTR_REPORT,DATA_REPORT_SHA256=$SCTR_REPORT_SHA256,SPLIT=development,CANDIDATES_OUTPUT=$candidates,REPORT=$shard_report,SHARD_INDEX=$index,SHARD_COUNT=$SHARDS" \
      "$RUNTIME/train/jobs/ttr1_control_evaluate.sbatch")
    jobs+=("$job")
    all_eval_jobs+=("$job")
    reports+=("$shard_report")
  done
  local dependency reports_csv merge
  dependency=$(IFS=:; echo "${jobs[*]}")
  reports_csv=$(IFS=:; echo "${reports[*]}")
  merge=$(submit --dependency=afterok:"$dependency" \
    --export=ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,DATA=$SCTR_DEV,DATA_SHA256=$SCTR_DEV_SHA256,SPLIT=development,SHARD_REPORTS=$reports_csv,CANDIDATES_OUTPUT=$root/candidates.jsonl,REPORT=$root/report.json \
    "$RUNTIME/pipeline/jobs/merge_ttr1_control_shards.sbatch")
  merge_jobs+=("$merge")
  LAST_MERGE=$merge
}

submit_trained_arm selective sctr1 "$selective_fit" "$OUTPUT_ROOT/selective_fit/checkpoint_0000256.pt" "$SCTR_DEV" "$SCTR_DEV_SHA256" "$SCTR_REPORT" "$SCTR_REPORT_SHA256" false
selective_merge=$LAST_MERGE
submit_trained_arm shuffled sctr1 "$shuffled_fit" "$OUTPUT_ROOT/shuffled_fit/checkpoint_0000256.pt" "$SCTR_DEV" "$SCTR_DEV_SHA256" "$SCTR_REPORT" "$SCTR_REPORT_SHA256" false
shuffled_merge=$LAST_MERGE
submit_trained_arm independent_commitment sctr1 "$independent_fit" "$OUTPUT_ROOT/independent_fit/checkpoint_0000256.pt" "$SCTR_DEV" "$SCTR_DEV_SHA256" "$SCTR_REPORT" "$SCTR_REPORT_SHA256" true
independent_merge=$LAST_MERGE
submit_trained_arm always_revise idr1 "$always_fit" "$OUTPUT_ROOT/always_revise_fit/checkpoint_0000256.pt" "$STANDARD_DEV" "$STANDARD_DEV_SHA256" "$STANDARD_REPORT" "$STANDARD_REPORT_SHA256" false
always_merge=$LAST_MERGE
submit_base_control unchanged_second_pass
unchanged_merge=$LAST_MERGE
submit_base_control long_single_generation
long_merge=$LAST_MERGE

comparison_dependency=$selective_merge:$shuffled_merge:$independent_merge:$always_merge:$unchanged_merge:$long_merge
comparison_job=$(submit --dependency=afterok:"$comparison_dependency" \
  --export=ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,SELECTIVE_REPORT=$OUTPUT_ROOT/development/selective/report.json,ALWAYS_REVISE_REPORT=$OUTPUT_ROOT/development/always_revise/report.json,UNCHANGED_REPORT=$OUTPUT_ROOT/development/unchanged_second_pass/report.json,LONG_REPORT=$OUTPUT_ROOT/development/long_single_generation/report.json,INDEPENDENT_REPORT=$OUTPUT_ROOT/development/independent_commitment/report.json,SHUFFLED_REPORT=$OUTPUT_ROOT/development/shuffled/report.json,SCTR_DATA_REPORT=$SCTR_REPORT,STANDARD_DATA_REPORT=$STANDARD_REPORT,OUTPUT=$OUTPUT_ROOT/development/comparison.json \
  "$RUNTIME/pipeline/jobs/compare_sctr1_development.sbatch")

temporary=$DISPATCH_RECEIPT.partial.$$
eval_jobs_csv=$(IFS=,; echo "${all_eval_jobs[*]}")
merge_jobs_csv=$(IFS=,; echo "${merge_jobs[*]}")
"$PYTHON" - "$temporary" <<PY
import json
import os

payload = {
    "comparison_job": "$comparison_job",
    "evaluation_jobs": "$eval_jobs_csv".split(","),
    "fit_jobs": {
        "always_revise": "$always_fit",
        "independent_commitment": "$independent_fit",
        "selective": "$selective_fit",
        "shuffled": "$shuffled_fit",
    },
    "merge_jobs": "$merge_jobs_csv".split(","),
    "runtime_manifest_sha256": "$RUNTIME_MANIFEST_SHA256",
    "shards_per_arm": $SHARDS,
    "status": "submitted",
}
with open(os.path.abspath("$temporary"), "x", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True)
    handle.write("\n")
PY
mv "$temporary" "$DISPATCH_RECEIPT"
chmod a-w "$DISPATCH_RECEIPT"
cat "$DISPATCH_RECEIPT"
