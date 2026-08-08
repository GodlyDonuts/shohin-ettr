#!/bin/bash
#SBATCH --job-name=idr-scale-train
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/idr_scale_train_dispatch_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/idr_scale_train_dispatch_%j.err

set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

: "${RUNTIME:?RUNTIME is required}"
: "${RUNTIME_MANIFEST_SHA256:?RUNTIME_MANIFEST_SHA256 is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${MODEL_REVISION:?MODEL_REVISION is required}"
: "${MODEL_CONFIG_SHA256:?MODEL_CONFIG_SHA256 is required}"
: "${WARM_START:?WARM_START is required}"
: "${WARM_START_SHA256:?WARM_START_SHA256 is required}"
: "${DATA_ROOT:?DATA_ROOT is required}"
: "${TRAIN_OUTPUT:?TRAIN_OUTPUT is required}"
: "${TRAINED_EVAL_ROOT:?TRAINED_EVAL_ROOT is required}"
: "${CONTROL_EVAL_ROOT:?CONTROL_EVAL_ROOT is required}"
: "${DISPATCH_ROOT:?DISPATCH_ROOT is required}"
SHARD_COUNT=${SHARD_COUNT:-4}
EXCLUDE_NODES=${EXCLUDE_NODES:-evc33:evc38}
EXCLUDE=${EXCLUDE_NODES//:/,}

test ! -e "$DISPATCH_ROOT"
test ! -e "$TRAIN_OUTPUT"
test -x "$PYTHON"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$WARM_START" | cut -d' ' -f1)" = "$WARM_START_SHA256"

DATA=$DATA_ROOT/train.jsonl
DATA_SHA256=$(sha256sum "$DATA" | cut -d' ' -f1)
DATA_REPORT=$DATA_ROOT/report.json
DATA_REPORT_SHA256=$(sha256sum "$DATA_REPORT" | cut -d' ' -f1)
"$PYTHON" - "$DATA" "$DATA_REPORT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

data, report_path = map(Path, sys.argv[1:])
report = json.loads(report_path.read_text(encoding="utf-8"))
expected = report.get("outputs", {}).get("train", {})
if report.get("schema") != "shohin-idr1-revision-data-report-v1":
    raise SystemExit("IDR scale report schema differs")
if expected.get("rows") != 9655:
    raise SystemExit("IDR scale train cardinality differs")
if Path(expected.get("path", "")).resolve() != data.resolve():
    raise SystemExit("IDR scale train path differs")
if expected.get("sha256") != hashlib.sha256(data.read_bytes()).hexdigest():
    raise SystemExit("IDR scale train hash differs")
PY

mkdir -p "$DISPATCH_ROOT"
train_exports="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
train_exports+=",MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION"
train_exports+=",MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256"
train_exports+=",WARM_START=$WARM_START,WARM_START_SHA256=$WARM_START_SHA256"
train_exports+=",DATA=$DATA,DATA_SHA256=$DATA_SHA256"
train_exports+=",DATA_REPORT=$DATA_REPORT,DATA_REPORT_SHA256=$DATA_REPORT_SHA256"
train_exports+=",OUTPUT=$TRAIN_OUTPUT"
train_job=$(sbatch --parsable --exclude="$EXCLUDE" --export="$train_exports" \
  "$RUNTIME/train/jobs/hf_idr1_train_reviser.sbatch")

trained_checkpoint=$TRAIN_OUTPUT/checkpoint_0000256.pt
trained_exports="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
trained_exports+=",MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION"
trained_exports+=",MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256"
trained_exports+=",ADAPTER_CHECKPOINT=$trained_checkpoint,DATA_ROOT=$DATA_ROOT"
trained_exports+=",EVAL_ROOT=$TRAINED_EVAL_ROOT"
trained_exports+=",DISPATCH_RECEIPT=$DISPATCH_ROOT/trained_eval_dispatch.json"
trained_exports+=",SHARD_COUNT=$SHARD_COUNT,EXCLUDE_NODES=$EXCLUDE_NODES"
trained_dispatch=$(sbatch --parsable --dependency=afterok:"$train_job" \
  --export="$trained_exports" "$RUNTIME/train/jobs/dispatch_idr_scale_evals.sbatch")

control_exports="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
control_exports+=",MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION"
control_exports+=",MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256"
control_exports+=",ADAPTER_CHECKPOINT=$WARM_START,ADAPTER_CHECKPOINT_SHA256=$WARM_START_SHA256"
control_exports+=",DATA_ROOT=$DATA_ROOT,EVAL_ROOT=$CONTROL_EVAL_ROOT"
control_exports+=",DISPATCH_RECEIPT=$DISPATCH_ROOT/control_eval_dispatch.json"
control_exports+=",SHARD_COUNT=$SHARD_COUNT,EXCLUDE_NODES=$EXCLUDE_NODES"
control_dispatch=$(sbatch --parsable --export="$control_exports" \
  "$RUNTIME/train/jobs/dispatch_idr_scale_evals.sbatch")

compare_dispatch=""
if [[ -n "${COMPARISON_OUTPUT:-}" ]]; then
  compare_exports="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
  compare_exports+=",TRAINED_DISPATCH_RECEIPT=$DISPATCH_ROOT/trained_eval_dispatch.json"
  compare_exports+=",CONTROL_DISPATCH_RECEIPT=$DISPATCH_ROOT/control_eval_dispatch.json"
  compare_exports+=",TRAINED_EVAL_ROOT=$TRAINED_EVAL_ROOT,CONTROL_EVAL_ROOT=$CONTROL_EVAL_ROOT"
  compare_exports+=",COMPARISON_OUTPUT=$COMPARISON_OUTPUT"
  compare_exports+=",DISPATCH_RECEIPT=$DISPATCH_ROOT/compare_dispatch.json"
  compare_dispatch=$(sbatch --parsable \
    --dependency=afterok:"$trained_dispatch":"$control_dispatch" \
    --export="$compare_exports" "$RUNTIME/train/jobs/dispatch_idr_scale_compare.sbatch")
fi

receipt=$DISPATCH_ROOT/train_dispatch.json
temporary=$receipt.partial.$$
"$PYTHON" - "$temporary" "$train_job" "$trained_dispatch" "$control_dispatch" \
  "$compare_dispatch" "$DATA_SHA256" "$DATA_REPORT_SHA256" <<'PY'
import json
from pathlib import Path
import sys

path, train, trained, control, compare, data_sha, report_sha = sys.argv[1:]
Path(path).write_text(
    json.dumps(
        {
            "status": "submitted",
            "train_job": train,
            "trained_eval_dispatch_job": trained,
            "control_eval_dispatch_job": control,
            "compare_dispatch_job": compare or None,
            "data_sha256": data_sha,
            "data_report_sha256": report_sha,
        },
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
mv "$temporary" "$receipt"
chmod a-w "$receipt"
printf 'train_job=%s trained_eval_dispatch=%s control_eval_dispatch=%s compare_dispatch=%s\n' \
  "$train_job" "$trained_dispatch" "$control_dispatch" "$compare_dispatch"
