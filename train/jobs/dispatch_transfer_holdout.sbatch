#!/bin/bash
#SBATCH --job-name=transfer-holdout-dispatch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/transfer_holdout_dispatch_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/transfer_holdout_dispatch_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION \
  MODEL_CONFIG_SHA256 MODEL_MANIFEST_SHA256 DATA_ROOT TREATMENT_FIT \
  INDEPENDENT_FIT DEVELOPMENT_COMPARISON OUTPUT_ROOT DISPATCH_RECEIPT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
SHARDS=${SHARDS:-8}; EXCLUDE=${EXCLUDE:-evc33,evc38,evc46}
test "$SHARDS" -eq 8; test ! -e "$OUTPUT_ROOT"; test ! -e "$DISPATCH_RECEIPT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$MODEL_ROOT/SHA256SUMS" | cut -d' ' -f1)" = "$MODEL_MANIFEST_SHA256"
(cd "$MODEL_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
python3 - "$DEVELOPMENT_COMPARISON" <<'PY'
import json, sys
value=json.load(open(sys.argv[1], encoding="utf-8"))
if (value.get("schema") != "shohin-ttr1-development-comparison-v1"
        or value.get("gate_pass") is not True
        or value.get("holdout_authorized") is not True):
    raise SystemExit("development comparison did not authorize holdout")
PY

DATA=$DATA_ROOT/holdout.jsonl; DATA_REPORT=$DATA_ROOT/report.json
TREATMENT_CHECKPOINT=$TREATMENT_FIT/checkpoint_0000256.pt
INDEPENDENT_CHECKPOINT=$INDEPENDENT_FIT/checkpoint_0000256.pt
for path in "$DATA" "$DATA_REPORT" "$TREATMENT_CHECKPOINT" \
  "$INDEPENDENT_CHECKPOINT"; do test -f "$path"; done
DATA_SHA256=$(sha256sum "$DATA" | cut -d' ' -f1)
DATA_REPORT_SHA256=$(sha256sum "$DATA_REPORT" | cut -d' ' -f1)
TREATMENT_SHA256=$(sha256sum "$TREATMENT_CHECKPOINT" | cut -d' ' -f1)
INDEPENDENT_SHA256=$(sha256sum "$INDEPENDENT_CHECKPOINT" | cut -d' ' -f1)
mkdir -p "$OUTPUT_ROOT" "$(dirname "$DISPATCH_RECEIPT")"
submit() { local raw; raw=$(sbatch --parsable "$@"); printf '%s\n' "${raw%%;*}"; }
common="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
common+=",MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION"
common+=",MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256,MODEL_MANIFEST_SHA256=$MODEL_MANIFEST_SHA256"
common+=",DATA=$DATA,DATA_SHA256=$DATA_SHA256,DATA_REPORT=$DATA_REPORT"
common+=",DATA_REPORT_SHA256=$DATA_REPORT_SHA256,SPLIT=holdout"
all_shards=(); merges=()

submit_arm() {
  local name=$1 kind=$2 checkpoint=${3:-} checkpoint_sha=${4:-}
  local root=$OUTPUT_ROOT/$name; mkdir -p "$root/shards"
  local jobs=() reports=()
  for index in $(seq 0 $((SHARDS - 1))); do
    local suffix candidates report exports job
    suffix=$(printf '%02d' "$index")
    candidates=$root/shards/candidates_$suffix.jsonl
    report=$root/shards/report_$suffix.json
    exports="$common,SHARD_INDEX=$index,SHARD_COUNT=$SHARDS"
    exports+=",CANDIDATES_OUTPUT=$candidates,REPORT=$report"
    if [[ "$kind" == treatment ]]; then
      exports+=",ADAPTER_CHECKPOINT=$checkpoint,ADAPTER_CHECKPOINT_SHA256=$checkpoint_sha"
      job=$(submit --exclude="$EXCLUDE" --export="$exports" \
        "$RUNTIME/train/jobs/hf_idr1_evaluate_reviser.sbatch")
    else
      exports+=",CONTROL=$name"
      if [[ "$kind" == independent ]]; then
        exports+=",ADAPTER_CHECKPOINT=$checkpoint,ADAPTER_CHECKPOINT_SHA256=$checkpoint_sha"
      fi
      job=$(submit --exclude="$EXCLUDE" --export="$exports" \
        "$RUNTIME/train/jobs/ttr1_control_evaluate.sbatch")
    fi
    jobs+=("$job"); all_shards+=("$job"); reports+=("$report")
  done
  local dependency report_csv merge
  dependency=$(IFS=:; echo "${jobs[*]}"); report_csv=$(IFS=:; echo "${reports[*]}")
  if [[ "$kind" == treatment ]]; then
    merge=$(submit --dependency=afterok:"$dependency" \
      --export="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,EVALUATOR=idr1,DATA=$DATA,DATA_SHA256=$DATA_SHA256,SPLIT=holdout,SHARD_REPORTS=$report_csv,CANDIDATES_OUTPUT=$root/candidates.jsonl,REPORT=$root/report.json" \
      "$RUNTIME/pipeline/jobs/merge_temporal_evaluation_shards.sbatch")
  else
    merge=$(submit --dependency=afterok:"$dependency" \
      --export="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,DATA=$DATA,DATA_SHA256=$DATA_SHA256,SPLIT=holdout,SHARD_REPORTS=$report_csv,CANDIDATES_OUTPUT=$root/candidates.jsonl,REPORT=$root/report.json" \
      "$RUNTIME/pipeline/jobs/merge_ttr1_control_shards.sbatch")
  fi
  merges+=("$merge"); LAST_MERGE=$merge
}

submit_arm treatment treatment "$TREATMENT_CHECKPOINT" "$TREATMENT_SHA256"
submit_arm unchanged_second_pass base
submit_arm self_refinement base
submit_arm long_single_generation base
submit_arm best_of_two base
submit_arm independent_commitment independent "$INDEPENDENT_CHECKPOINT" "$INDEPENDENT_SHA256"
dependency=$(IFS=:; echo "${merges[*]}")
comparison_job=$(submit --dependency=afterok:"$dependency" \
  --export="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,DEVELOPMENT_COMPARISON=$DEVELOPMENT_COMPARISON,TREATMENT_REPORT=$OUTPUT_ROOT/treatment/report.json,UNCHANGED_REPORT=$OUTPUT_ROOT/unchanged_second_pass/report.json,SELF_REFINEMENT_REPORT=$OUTPUT_ROOT/self_refinement/report.json,LONG_SINGLE_REPORT=$OUTPUT_ROOT/long_single_generation/report.json,BEST_OF_TWO_REPORT=$OUTPUT_ROOT/best_of_two/report.json,INDEPENDENT_REPORT=$OUTPUT_ROOT/independent_commitment/report.json,OUTPUT=$OUTPUT_ROOT/comparison.json" \
  "$RUNTIME/pipeline/jobs/compare_ttr1_holdout.sbatch")
temporary=$DISPATCH_RECEIPT.partial.$$
shard_csv=$(IFS=,; echo "${all_shards[*]}"); merge_csv=$(IFS=,; echo "${merges[*]}")
cat >"$temporary" <<EOF
{"comparison_job":"$comparison_job","development_comparison_sha256":"$(sha256sum "$DEVELOPMENT_COMPARISON" | cut -d' ' -f1)","merge_jobs":"$merge_csv","model_manifest_sha256":"$MODEL_MANIFEST_SHA256","runtime_manifest_sha256":"$RUNTIME_MANIFEST_SHA256","shard_jobs":"$shard_csv","status":"submitted"}
EOF
mv "$temporary" "$DISPATCH_RECEIPT"; chmod a-w "$DISPATCH_RECEIPT"
cat "$DISPATCH_RECEIPT"
