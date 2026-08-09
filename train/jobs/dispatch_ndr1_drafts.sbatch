#!/bin/bash
#SBATCH --job-name=ndr1-dispatch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/ndr1_dispatch_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/ndr1_dispatch_%j.err

set -euo pipefail

for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION \
  MODEL_CONFIG_SHA256 MODEL_MANIFEST_SHA256 ADAPTER_CHECKPOINT \
  ADAPTER_CHECKPOINT_SHA256 SOURCE SOURCE_REPORT DRAFT_ROOT DATA_OUTPUT \
  RECEIPT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
EXCLUDE_NODES=${EXCLUDE_NODES:-evc26:evc31:evc33:evc38:evc46}
EXCLUDE=${EXCLUDE_NODES//:/,}
SHARD_COUNT=16
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=$BASE/envs/product-reasoning-b3a3603-r2/bin/python

test ! -e "$DRAFT_ROOT"
test ! -e "$DATA_OUTPUT"
test ! -e "$RECEIPT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = \
  "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
source_sha256=$(sha256sum "$SOURCE" | cut -d' ' -f1)
source_report_sha256=$(sha256sum "$SOURCE_REPORT" | cut -d' ' -f1)
mkdir -p "$DRAFT_ROOT" "$(dirname "$RECEIPT")"

jobs=()
report_args=""
for shard in $(seq 0 $((SHARD_COUNT - 1))); do
  output=$DRAFT_ROOT/shard_$shard.jsonl
  report=$DRAFT_ROOT/shard_$shard.report.json
  exports="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
  exports+=",MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION"
  exports+=",MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256"
  exports+=",MODEL_MANIFEST_SHA256=$MODEL_MANIFEST_SHA256"
  exports+=",ADAPTER_CHECKPOINT=$ADAPTER_CHECKPOINT"
  exports+=",ADAPTER_CHECKPOINT_SHA256=$ADAPTER_CHECKPOINT_SHA256"
  exports+=",SOURCE=$SOURCE,SOURCE_SHA256=$source_sha256"
  exports+=",SOURCE_REPORT=$SOURCE_REPORT,SOURCE_REPORT_SHA256=$source_report_sha256"
  exports+=",OUTPUT=$output,REPORT=$report,SHARD_INDEX=$shard,SHARD_COUNT=$SHARD_COUNT"
  job=$(sbatch --parsable --exclude="$EXCLUDE" --export="$exports" \
    "$RUNTIME/train/jobs/hf_ndr1_generate_drafts.sbatch")
  jobs+=("$job")
  report_args+=" --draft-report $report"
done

dependency=$(IFS=:; echo "${jobs[*]}")
merge_job=$(sbatch --parsable --dependency=afterok:"$dependency" \
  --partition=normal --time=02:00:00 --cpus-per-task=4 --mem=96G \
  --wrap="set -euo pipefail; test \$(sha256sum $RUNTIME/SHA256SUMS | cut -d\\  -f1) = $RUNTIME_MANIFEST_SHA256; export PYTHONPATH=$RUNTIME/train:$RUNTIME/pipeline; $PYTHON $RUNTIME/pipeline/build_ndr1_natural_revision_data.py --source $SOURCE --source-report $SOURCE_REPORT $report_args --adapter-checkpoint-sha256 $ADAPTER_CHECKPOINT_SHA256 --model-root $MODEL_ROOT --output $DATA_OUTPUT; sha256sum $DATA_OUTPUT/train_aligned.jsonl $DATA_OUTPUT/train_shuffled.jsonl $DATA_OUTPUT/report.json; chmod -R a-w $DATA_OUTPUT")

temporary=$RECEIPT.partial.$$
"$PYTHON" - "$temporary" "${jobs[*]}" "$merge_job" "$source_sha256" <<'PY'
import json
from pathlib import Path
import sys

path, jobs, merge_job, source_sha256 = sys.argv[1:]
Path(path).write_text(
    json.dumps(
        {
            "draft_jobs": jobs.split(),
            "merge_job": merge_job,
            "source_sha256": source_sha256,
            "status": "submitted",
        },
        sort_keys=True,
    )
    + "\n"
)
PY
mv "$temporary" "$RECEIPT"
chmod a-w "$RECEIPT"
cat "$RECEIPT"
