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
mkdir -p "$OUTPUT_ROOT" "$(dirname "$DISPATCH_RECEIPT")"

common="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
common+=",MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION"
common+=",MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256,MODEL_LOADER=auto"
common+=",DATA=$DATA,DATA_SHA256=$DATA_SHA256"
common+=",DATA_REPORT=$DATA_REPORT,DATA_REPORT_SHA256=$DATA_REPORT_SHA256"
common+=",SPLIT=holdout"

submit_control() {
  local name=$1
  local checkpoint=$2
  local checkpoint_sha256=$3
  local root=$OUTPUT_ROOT/$name
  mkdir -p "$root"
  sbatch --parsable --exclude=evc33,evc38 \
    --export="$common,CONTROL=$name,ADAPTER_CHECKPOINT=$checkpoint,ADAPTER_CHECKPOINT_SHA256=$checkpoint_sha256,CANDIDATES_OUTPUT=$root/candidates.jsonl,REPORT=$root/report.json" \
    "$RUNTIME/train/jobs/ttr1_control_evaluate.sbatch"
}

treatment_root=$OUTPUT_ROOT/treatment
mkdir -p "$treatment_root"
treatment_job=$(sbatch --parsable --exclude=evc33,evc38 \
  --export="$common,ADAPTER_CHECKPOINT=$TREATMENT_CHECKPOINT,ADAPTER_CHECKPOINT_SHA256=$TREATMENT_CHECKPOINT_SHA256,CANDIDATES_OUTPUT=$treatment_root/candidates.jsonl,REPORT=$treatment_root/report.json" \
  "$RUNTIME/train/jobs/hf_idr1_evaluate_reviser.sbatch")
unchanged_job=$(submit_control unchanged_second_pass "$WARM_START" "$WARM_START_SHA256")
self_job=$(submit_control self_refinement "$WARM_START" "$WARM_START_SHA256")
long_job=$(submit_control long_single_generation "$WARM_START" "$WARM_START_SHA256")
best_job=$(submit_control best_of_two "$WARM_START" "$WARM_START_SHA256")
independent_job=$(submit_control independent_commitment "$INDEPENDENT_CHECKPOINT" "$INDEPENDENT_CHECKPOINT_SHA256")

temporary=$DISPATCH_RECEIPT.partial.$$
cat >"$temporary" <<EOF
{"best_of_two_job":"$best_job","development_comparison":"$DEVELOPMENT_COMPARISON","development_comparison_sha256":"$development_sha256","independent_commitment_job":"$independent_job","long_single_job":"$long_job","runtime_manifest_sha256":"$RUNTIME_MANIFEST_SHA256","self_refinement_job":"$self_job","status":"submitted","treatment_job":"$treatment_job","unchanged_second_pass_job":"$unchanged_job"}
EOF
mv "$temporary" "$DISPATCH_RECEIPT"
chmod a-w "$DISPATCH_RECEIPT"
cat "$DISPATCH_RECEIPT"
