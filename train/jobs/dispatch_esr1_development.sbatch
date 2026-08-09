#!/bin/bash
#SBATCH --job-name=esr1-dispatch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/esr1_dispatch_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/esr1_dispatch_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION \
  MODEL_CONFIG_SHA256 STANDARD_ROOT ALWAYS_REPORT OUTPUT_ROOT DISPATCH_RECEIPT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test ! -e "$OUTPUT_ROOT"; test ! -e "$DISPATCH_RECEIPT"; test -f "$ALWAYS_REPORT"
for name in train development report; do
  suffix=jsonl; [[ "$name" == report ]] && suffix=json
  test -f "$STANDARD_ROOT/$name.$suffix"
done

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
TRAIN=$STANDARD_ROOT/train.jsonl
DEV=$STANDARD_ROOT/development.jsonl
DATA_REPORT=$STANDARD_ROOT/report.json
TRAIN_SHA=$(sha256sum "$TRAIN" | cut -d' ' -f1)
DEV_SHA=$(sha256sum "$DEV" | cut -d' ' -f1)
DATA_REPORT_SHA=$(sha256sum "$DATA_REPORT" | cut -d' ' -f1)
SHARDS=8
EXCLUDE=${EXCLUDE:-evc33,evc38,evc46}
mkdir -p "$OUTPUT_ROOT" "$(dirname "$DISPATCH_RECEIPT")"

submit() { local raw; raw=$(sbatch --parsable "$@"); printf '%s\n' "${raw%%;*}"; }
common="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
common+=",MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION"
common+=",MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256,DATA_REPORT=$DATA_REPORT"
common+=",DATA_REPORT_SHA256=$DATA_REPORT_SHA"

syndrome_fit=$(submit --exclude="$EXCLUDE" \
  --export="$common,DATA=$TRAIN,DATA_SHA256=$TRAIN_SHA,OUTPUT=$OUTPUT_ROOT/syndrome_fit,ARM=syndrome" \
  "$RUNTIME/train/jobs/hf_esr1_train.sbatch")
ettr_fit=$(submit --exclude="$EXCLUDE" \
  --export="$common,DATA=$TRAIN,DATA_SHA256=$TRAIN_SHA,OUTPUT=$OUTPUT_ROOT/workspace_control_fit,ARM=ettr" \
  "$RUNTIME/train/jobs/hf_esr1_train.sbatch")

release=$(submit --dependency=afterok:"$syndrome_fit:$ettr_fit" \
  --export="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION,MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256,STANDARD_ROOT=$STANDARD_ROOT,ALWAYS_REPORT=$ALWAYS_REPORT,OUTPUT_ROOT=$OUTPUT_ROOT,RELEASE_RECEIPT=$OUTPUT_ROOT/evaluation_dispatch.json" \
  "$RUNTIME/train/jobs/dispatch_esr1_evaluation.sbatch")

temporary=$DISPATCH_RECEIPT.partial.$$
"$PYTHON" - "$temporary" <<PY
import json, os
payload = {
  "fit_jobs": {"syndrome": "$syndrome_fit", "workspace_control": "$ettr_fit"},
  "evaluation_release_job": "$release",
  "runtime_manifest_sha256": "$RUNTIME_MANIFEST_SHA256",
  "status": "submitted",
}
with open(os.path.abspath("$temporary"), "x", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True); handle.write("\n")
PY
mv "$temporary" "$DISPATCH_RECEIPT"; chmod a-w "$DISPATCH_RECEIPT"
cat "$DISPATCH_RECEIPT"
