#!/bin/bash
#SBATCH --job-name=idr1-eval-dispatch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/idr1_eval_dispatch_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/idr1_eval_dispatch_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${RUNTIME_MANIFEST_SHA256:?RUNTIME_MANIFEST_SHA256 is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${MODEL_REVISION:?MODEL_REVISION is required}"
: "${MODEL_CONFIG_SHA256:?MODEL_CONFIG_SHA256 is required}"
: "${TRAIN_OUTPUT:?TRAIN_OUTPUT is required}"
: "${DATA_ROOT:?DATA_ROOT is required}"
: "${EVAL_ROOT:?EVAL_ROOT is required}"
: "${DISPATCH_RECEIPT:?DISPATCH_RECEIPT is required}"

test ! -e "$DISPATCH_RECEIPT"
test -f "$TRAIN_OUTPUT/checkpoint_0000256.pt"
test -f "$TRAIN_OUTPUT/report.json"
test -f "$DATA_ROOT/report.json"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"

CHECKPOINT=$TRAIN_OUTPUT/checkpoint_0000256.pt
CHECKPOINT_SHA256=$(sha256sum "$CHECKPOINT" | cut -d' ' -f1)
DATA_REPORT=$DATA_ROOT/report.json
DATA_REPORT_SHA256=$(sha256sum "$DATA_REPORT" | cut -d' ' -f1)
mkdir -p "$EVAL_ROOT" "$(dirname "$DISPATCH_RECEIPT")"

submit_eval() {
  local split=$1
  local data=$DATA_ROOT/$split.jsonl
  local data_sha256
  data_sha256=$(sha256sum "$data" | cut -d' ' -f1)
  local output=$EVAL_ROOT/$split
  local exports
  exports="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
  exports+=",MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION"
  exports+=",MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256"
  exports+=",ADAPTER_CHECKPOINT=$CHECKPOINT,ADAPTER_CHECKPOINT_SHA256=$CHECKPOINT_SHA256"
  exports+=",DATA=$data,DATA_SHA256=$data_sha256"
  exports+=",DATA_REPORT=$DATA_REPORT,DATA_REPORT_SHA256=$DATA_REPORT_SHA256"
  exports+=",SPLIT=$split,CANDIDATES_OUTPUT=$output/candidates.jsonl"
  exports+=",REPORT=$output/report.json"
  sbatch --parsable --exclude=evc33 --export="$exports" \
    "$RUNTIME/train/jobs/hf_idr1_evaluate_reviser.sbatch"
}

development_job=$(submit_eval development)
holdout_job=$(submit_eval holdout)
temporary=$DISPATCH_RECEIPT.partial.$$
cat >"$temporary" <<EOF
{"checkpoint_sha256":"$CHECKPOINT_SHA256","development_job":"$development_job","holdout_job":"$holdout_job","runtime_manifest_sha256":"$RUNTIME_MANIFEST_SHA256","status":"submitted"}
EOF
mv "$temporary" "$DISPATCH_RECEIPT"
chmod a-w "$DISPATCH_RECEIPT"
printf 'development_job=%s holdout_job=%s\n' "$development_job" "$holdout_job"
