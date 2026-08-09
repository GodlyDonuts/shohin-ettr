#!/bin/bash
#SBATCH --job-name=ttr1-holdout-dispatch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/ttr1_holdout_dispatch_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/ttr1_holdout_dispatch_%j.err

set -euo pipefail

for variable in RUNTIME RUNTIME_MANIFEST_SHA256 DEVELOPMENT_COMPARISON \
  MODEL_ROOT MODEL_REVISION MODEL_CONFIG_SHA256 WARM_START WARM_START_SHA256 \
  TREATMENT_CHECKPOINT TREATMENT_CHECKPOINT_SHA256 INDEPENDENT_CHECKPOINT \
  INDEPENDENT_CHECKPOINT_SHA256 DATA_ROOT OUTPUT_ROOT DISPATCH_RECEIPT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$WARM_START" | cut -d' ' -f1)" = "$WARM_START_SHA256"
test "$(sha256sum "$TREATMENT_CHECKPOINT" | cut -d' ' -f1)" = "$TREATMENT_CHECKPOINT_SHA256"
test "$(sha256sum "$INDEPENDENT_CHECKPOINT" | cut -d' ' -f1)" = "$INDEPENDENT_CHECKPOINT_SHA256"
test -f "$DATA_ROOT/holdout.jsonl"
test -f "$DATA_ROOT/report.json"
test ! -e "$OUTPUT_ROOT"
test ! -e "$DISPATCH_RECEIPT"

development_sha256=$(sha256sum "$DEVELOPMENT_COMPARISON" | cut -d' ' -f1)
python3 - "$DEVELOPMENT_COMPARISON" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    value.get("schema") != "shohin-ttr1-development-comparison-v1"
    or value.get("status") != "complete"
    or value.get("gate_pass") is not True
    or value.get("holdout_authorized") is not True
):
    raise SystemExit("development comparison did not authorize holdout")
PY

DATA=$DATA_ROOT/holdout.jsonl
DATA_SHA256=$(sha256sum "$DATA" | cut -d' ' -f1)
DATA_REPORT=$DATA_ROOT/report.json
DATA_REPORT_SHA256=$(sha256sum "$DATA_REPORT" | cut -d' ' -f1)
CONTROL_SHARDS=${CONTROL_SHARDS:-8}
test "$CONTROL_SHARDS" -eq 8
mkdir -p "$OUTPUT_ROOT" "$(dirname "$DISPATCH_RECEIPT")"

common="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
common+=",MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION"
common+=",MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256,MODEL_LOADER=auto"
common+=",DATA=$DATA,DATA_SHA256=$DATA_SHA256"
common+=",DATA_REPORT=$DATA_REPORT,DATA_REPORT_SHA256=$DATA_REPORT_SHA256"
common+=",SPLIT=holdout"

treatment_root=$OUTPUT_ROOT/treatment
mkdir -p "$treatment_root"
treatment_job=$(sbatch --parsable --exclude=evc33,evc38 \
  --export="$common,ADAPTER_CHECKPOINT=$TREATMENT_CHECKPOINT,ADAPTER_CHECKPOINT_SHA256=$TREATMENT_CHECKPOINT_SHA256,CANDIDATES_OUTPUT=$treatment_root/candidates.jsonl,REPORT=$treatment_root/report.json" \
  "$RUNTIME/train/jobs/hf_idr1_evaluate_reviser.sbatch")

independent_root=$OUTPUT_ROOT/independent_commitment
mkdir -p "$independent_root"
independent_job=$(sbatch --parsable --exclude=evc33,evc38 \
  --export="$common,CONTROL=independent_commitment,ADAPTER_CHECKPOINT=$INDEPENDENT_CHECKPOINT,ADAPTER_CHECKPOINT_SHA256=$INDEPENDENT_CHECKPOINT_SHA256,CANDIDATES_OUTPUT=$independent_root/candidates.jsonl,REPORT=$independent_root/report.json" \
  "$RUNTIME/train/jobs/ttr1_control_evaluate.sbatch")

standard_controls=(
  unchanged_second_pass
  self_refinement
  long_single_generation
  best_of_two
)
all_shard_jobs=()
merge_jobs=()
for control in "${standard_controls[@]}"; do
  root=$OUTPUT_ROOT/$control
  mkdir -p "$root/shards"
  shard_jobs=()
  shard_reports=()
  for index in $(seq 0 $((CONTROL_SHARDS - 1))); do
    suffix=$(printf %02d "$index")
    candidates=$root/shards/candidates_$suffix.jsonl
    report=$root/shards/report_$suffix.json
    job=$(sbatch --parsable --exclude=evc33,evc38 \
      --export="$common,CONTROL=$control,ADAPTER_CHECKPOINT=$WARM_START,ADAPTER_CHECKPOINT_SHA256=$WARM_START_SHA256,CANDIDATES_OUTPUT=$candidates,REPORT=$report,SHARD_INDEX=$index,SHARD_COUNT=$CONTROL_SHARDS" \
      "$RUNTIME/train/jobs/ttr1_control_evaluate.sbatch")
    shard_jobs+=("$job")
    all_shard_jobs+=("$job")
    shard_reports+=("$report")
  done
  dependency=$(IFS=:; echo "${shard_jobs[*]}")
  reports_csv=$(IFS=:; echo "${shard_reports[*]}")
  merge=$(sbatch --parsable --dependency=afterok:"$dependency" \
    --export=ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,DATA=$DATA,DATA_SHA256=$DATA_SHA256,SPLIT=holdout,SHARD_REPORTS=$reports_csv,CANDIDATES_OUTPUT=$root/candidates.jsonl,REPORT=$root/report.json \
    "$RUNTIME/pipeline/jobs/merge_ttr1_control_shards.sbatch")
  merge_jobs+=("$merge")
done

comparison_dependency=$treatment_job:$independent_job:$(IFS=:; echo "${merge_jobs[*]}")
comparison_job=$(sbatch --parsable --dependency=afterok:"$comparison_dependency" \
  --export=ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,DEVELOPMENT_COMPARISON=$DEVELOPMENT_COMPARISON,TREATMENT_REPORT=$treatment_root/report.json,UNCHANGED_REPORT=$OUTPUT_ROOT/unchanged_second_pass/report.json,SELF_REFINEMENT_REPORT=$OUTPUT_ROOT/self_refinement/report.json,LONG_SINGLE_REPORT=$OUTPUT_ROOT/long_single_generation/report.json,BEST_OF_TWO_REPORT=$OUTPUT_ROOT/best_of_two/report.json,INDEPENDENT_REPORT=$independent_root/report.json,OUTPUT=$OUTPUT_ROOT/comparison.json \
  "$RUNTIME/pipeline/jobs/compare_ttr1_holdout.sbatch")

temporary=$DISPATCH_RECEIPT.partial.$$
shard_jobs_csv=$(IFS=,; echo "${all_shard_jobs[*]}")
merge_jobs_csv=$(IFS=,; echo "${merge_jobs[*]}")
python3 - "$temporary" <<PY
import json
import os

payload = {
    "comparison_job": "$comparison_job",
    "control_shards": $CONTROL_SHARDS,
    "development_comparison": "$DEVELOPMENT_COMPARISON",
    "development_comparison_sha256": "$development_sha256",
    "independent_commitment_job": "$independent_job",
    "merge_jobs": "$merge_jobs_csv".split(","),
    "runtime_manifest_sha256": "$RUNTIME_MANIFEST_SHA256",
    "shard_jobs": "$shard_jobs_csv".split(","),
    "status": "submitted",
    "treatment_job": "$treatment_job",
}
path = os.path.abspath("$temporary")
with open(path, "x", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True)
    handle.write("\n")
PY
mv "$temporary" "$DISPATCH_RECEIPT"
chmod a-w "$DISPATCH_RECEIPT"
cat "$DISPATCH_RECEIPT"
