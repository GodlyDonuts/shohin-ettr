#!/bin/bash
#SBATCH --job-name=ttr1-dev-dispatch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/ttr1_dispatch_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/ttr1_dispatch_%j.err

set -euo pipefail

for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION \
  MODEL_CONFIG_SHA256 WARM_START WARM_START_SHA256 DATA_ROOT OUTPUT_ROOT \
  DISPATCH_RECEIPT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$WARM_START" | cut -d' ' -f1)" = "$WARM_START_SHA256"
test -f "$DATA_ROOT/train.jsonl"
test -f "$DATA_ROOT/development.jsonl"
test -f "$DATA_ROOT/report.json"
test ! -e "$OUTPUT_ROOT"
test ! -e "$DISPATCH_RECEIPT"

TRAIN_DATA=$DATA_ROOT/train.jsonl
TRAIN_DATA_SHA256=$(sha256sum "$TRAIN_DATA" | cut -d' ' -f1)
DEV_DATA=$DATA_ROOT/development.jsonl
DEV_DATA_SHA256=$(sha256sum "$DEV_DATA" | cut -d' ' -f1)
DATA_REPORT=$DATA_ROOT/report.json
DATA_REPORT_SHA256=$(sha256sum "$DATA_REPORT" | cut -d' ' -f1)
mkdir -p "$OUTPUT_ROOT" "$(dirname "$DISPATCH_RECEIPT")"

common="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
common+=",MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION"
common+=",MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256,MODEL_LOADER=auto"
train_common="$common,WARM_START=$WARM_START,WARM_START_SHA256=$WARM_START_SHA256"
train_common+=",DATA=$TRAIN_DATA,DATA_SHA256=$TRAIN_DATA_SHA256"
train_common+=",DATA_REPORT=$DATA_REPORT,DATA_REPORT_SHA256=$DATA_REPORT_SHA256"

treatment_train=$(sbatch --parsable --exclude=evc33,evc38 \
  --export="$train_common,OUTPUT=$OUTPUT_ROOT/treatment_fit,MASK_INTERNAL_DRAFT=false" \
  "$RUNTIME/train/jobs/hf_idr1_train_reviser.sbatch")
independent_train=$(sbatch --parsable --exclude=evc33,evc38 \
  --export="$train_common,OUTPUT=$OUTPUT_ROOT/independent_fit,MASK_INTERNAL_DRAFT=true" \
  "$RUNTIME/train/jobs/hf_idr1_train_reviser.sbatch")

control_common="$common,ADAPTER_CHECKPOINT=$WARM_START"
control_common+=",ADAPTER_CHECKPOINT_SHA256=$WARM_START_SHA256"
control_common+=",DATA=$DEV_DATA,DATA_SHA256=$DEV_DATA_SHA256"
control_common+=",DATA_REPORT=$DATA_REPORT,DATA_REPORT_SHA256=$DATA_REPORT_SHA256"
control_common+=",SPLIT=development"

submit_control() {
  local name=$1
  local root=$OUTPUT_ROOT/development/$name
  mkdir -p "$root"
  sbatch --parsable --exclude=evc33,evc38 \
    --export="$control_common,CONTROL=$name,CANDIDATES_OUTPUT=$root/candidates.jsonl,REPORT=$root/report.json" \
    "$RUNTIME/train/jobs/ttr1_control_evaluate.sbatch"
}
unchanged_job=$(submit_control unchanged_second_pass)
self_job=$(submit_control self_refinement)
long_job=$(submit_control long_single_generation)
best_job=$(submit_control best_of_two)

treatment_checkpoint=$OUTPUT_ROOT/treatment_fit/checkpoint_0000256.pt
treatment_report=$OUTPUT_ROOT/development/treatment/report.json
mkdir -p "$(dirname "$treatment_report")"

# The checkpoint hash is produced by training, so tiny dependency dispatchers
# bind it after success instead of weakening the evaluation job's hash check.
submit_eval_dispatch() {
  local dependency=$1
  local checkpoint=$2
  local control=$3
  local output_root=$4
  local mode=$5
  sbatch --parsable --dependency=afterok:"$dependency" --wrap="set -euo pipefail
CHECKPOINT_SHA256=\$(sha256sum '$checkpoint' | cut -d' ' -f1)
if [[ '$mode' == treatment ]]; then
  sbatch --parsable --exclude=evc33,evc38 --export='$common,DATA=$DEV_DATA,DATA_SHA256=$DEV_DATA_SHA256,DATA_REPORT=$DATA_REPORT,DATA_REPORT_SHA256=$DATA_REPORT_SHA256,SPLIT=development',ADAPTER_CHECKPOINT='$checkpoint',ADAPTER_CHECKPOINT_SHA256=\$CHECKPOINT_SHA256,CANDIDATES_OUTPUT='$output_root/candidates.jsonl',REPORT='$output_root/report.json' '$RUNTIME/train/jobs/hf_idr1_evaluate_reviser.sbatch'
else
  sbatch --parsable --exclude=evc33,evc38 --export='$common,DATA=$DEV_DATA,DATA_SHA256=$DEV_DATA_SHA256,DATA_REPORT=$DATA_REPORT,DATA_REPORT_SHA256=$DATA_REPORT_SHA256,SPLIT=development',CONTROL='$control',ADAPTER_CHECKPOINT='$checkpoint',ADAPTER_CHECKPOINT_SHA256=\$CHECKPOINT_SHA256,CANDIDATES_OUTPUT='$output_root/candidates.jsonl',REPORT='$output_root/report.json' '$RUNTIME/train/jobs/ttr1_control_evaluate.sbatch'
fi" \
    --job-name=ttr1-eval-release --partition=normal --cpus-per-task=1 --mem=4G \
    --time=00:05:00 --output=/lustre/fs1/home/sa305415/shohin/logs/ttr1_release_%j.out
}
treatment_release=$(submit_eval_dispatch "$treatment_train" "$treatment_checkpoint" none "$OUTPUT_ROOT/development/treatment" treatment)
independent_checkpoint=$OUTPUT_ROOT/independent_fit/checkpoint_0000256.pt
independent_release=$(submit_eval_dispatch "$independent_train" "$independent_checkpoint" independent_commitment "$OUTPUT_ROOT/development/independent_commitment" control)

temporary=$DISPATCH_RECEIPT.partial.$$
cat >"$temporary" <<EOF
{"best_of_two_job":"$best_job","independent_release_job":"$independent_release","independent_train_job":"$independent_train","long_single_job":"$long_job","runtime_manifest_sha256":"$RUNTIME_MANIFEST_SHA256","self_refinement_job":"$self_job","status":"submitted","treatment_release_job":"$treatment_release","treatment_train_job":"$treatment_train","unchanged_second_pass_job":"$unchanged_job"}
EOF
mv "$temporary" "$DISPATCH_RECEIPT"
chmod a-w "$DISPATCH_RECEIPT"
cat "$DISPATCH_RECEIPT"
