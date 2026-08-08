#!/bin/bash
#SBATCH --job-name=idr1-train-dispatch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/idr1_train_dispatch_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/idr1_train_dispatch_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${RUNTIME_MANIFEST_SHA256:?RUNTIME_MANIFEST_SHA256 is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${MODEL_REVISION:?MODEL_REVISION is required}"
: "${MODEL_CONFIG_SHA256:?MODEL_CONFIG_SHA256 is required}"
: "${WARM_START:?WARM_START is required}"
: "${WARM_START_SHA256:?WARM_START_SHA256 is required}"
: "${DATA_ROOT:?DATA_ROOT is required}"
: "${TRAIN_OUTPUT:?TRAIN_OUTPUT is required}"
: "${EVAL_ROOT:?EVAL_ROOT is required}"
: "${DISPATCH_ROOT:?DISPATCH_ROOT is required}"

test ! -e "$DISPATCH_ROOT"
test ! -e "$TRAIN_OUTPUT"
test -f "$DATA_ROOT/train.jsonl"
test -f "$DATA_ROOT/report.json"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$WARM_START" | cut -d' ' -f1)" = "$WARM_START_SHA256"

DATA=$DATA_ROOT/train.jsonl
DATA_SHA256=$(sha256sum "$DATA" | cut -d' ' -f1)
DATA_REPORT=$DATA_ROOT/report.json
DATA_REPORT_SHA256=$(sha256sum "$DATA_REPORT" | cut -d' ' -f1)
python3 - "$DATA" "$DATA_REPORT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

data, report_path = map(Path, sys.argv[1:])
report = json.loads(report_path.read_text(encoding="utf-8"))
expected = report.get("outputs", {}).get("train", {})
if report.get("schema") != "shohin-idr1-revision-data-report-v1":
    raise SystemExit("IDR1 report schema differs")
if expected.get("rows") != 9655:
    raise SystemExit("IDR1 train cardinality differs")
if Path(expected.get("path", "")).resolve() != data.resolve():
    raise SystemExit("IDR1 train path differs")
if expected.get("sha256") != hashlib.sha256(data.read_bytes()).hexdigest():
    raise SystemExit("IDR1 train hash differs")
PY

mkdir -p "$DISPATCH_ROOT"
TRAIN_EXPORTS="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
TRAIN_EXPORTS+=",MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION"
TRAIN_EXPORTS+=",MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256"
TRAIN_EXPORTS+=",WARM_START=$WARM_START,WARM_START_SHA256=$WARM_START_SHA256"
TRAIN_EXPORTS+=",DATA=$DATA,DATA_SHA256=$DATA_SHA256"
TRAIN_EXPORTS+=",DATA_REPORT=$DATA_REPORT,DATA_REPORT_SHA256=$DATA_REPORT_SHA256"
TRAIN_EXPORTS+=",OUTPUT=$TRAIN_OUTPUT"
train_job=$(sbatch --parsable --exclude=evc33 --export="$TRAIN_EXPORTS" \
  "$RUNTIME/train/jobs/hf_idr1_train_reviser.sbatch")

EVAL_RECEIPT=$DISPATCH_ROOT/eval_dispatch.json
EVAL_EXPORTS="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
EVAL_EXPORTS+=",MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION"
EVAL_EXPORTS+=",MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256"
EVAL_EXPORTS+=",TRAIN_OUTPUT=$TRAIN_OUTPUT,DATA_ROOT=$DATA_ROOT,EVAL_ROOT=$EVAL_ROOT"
EVAL_EXPORTS+=",DISPATCH_RECEIPT=$EVAL_RECEIPT"
eval_dispatch_job=$(sbatch --parsable --dependency=afterok:"$train_job" \
  --export="$EVAL_EXPORTS" "$RUNTIME/train/jobs/dispatch_idr1_evals.sbatch")

receipt=$DISPATCH_ROOT/train_dispatch.json
temporary=$receipt.partial.$$
cat >"$temporary" <<EOF
{"data_report_sha256":"$DATA_REPORT_SHA256","data_sha256":"$DATA_SHA256","eval_dispatch_job":"$eval_dispatch_job","runtime_manifest_sha256":"$RUNTIME_MANIFEST_SHA256","status":"submitted","train_job":"$train_job"}
EOF
mv "$temporary" "$receipt"
chmod a-w "$receipt"
printf 'train_job=%s eval_dispatch_job=%s\n' "$train_job" "$eval_dispatch_job"
