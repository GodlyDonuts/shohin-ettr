#!/bin/bash
#SBATCH --job-name=idr-scale-evals
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/idr_scale_dispatch_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/idr_scale_dispatch_%j.err

set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

: "${RUNTIME:?RUNTIME is required}"
: "${RUNTIME_MANIFEST_SHA256:?RUNTIME_MANIFEST_SHA256 is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${MODEL_REVISION:?MODEL_REVISION is required}"
: "${MODEL_CONFIG_SHA256:?MODEL_CONFIG_SHA256 is required}"
: "${ADAPTER_CHECKPOINT:?ADAPTER_CHECKPOINT is required}"
: "${DATA_ROOT:?DATA_ROOT is required}"
: "${EVAL_ROOT:?EVAL_ROOT is required}"
: "${DISPATCH_RECEIPT:?DISPATCH_RECEIPT is required}"
SHARD_COUNT=${SHARD_COUNT:-4}
EXCLUDE=${EXCLUDE:-evc33,evc38}

test "$SHARD_COUNT" -ge 2
test -x "$PYTHON"
test ! -e "$DISPATCH_RECEIPT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
observed_checkpoint_sha256=$(sha256sum "$ADAPTER_CHECKPOINT" | cut -d' ' -f1)
if [[ -n "${ADAPTER_CHECKPOINT_SHA256:-}" ]]; then
  test "$observed_checkpoint_sha256" = "$ADAPTER_CHECKPOINT_SHA256"
else
  ADAPTER_CHECKPOINT_SHA256=$observed_checkpoint_sha256
fi

DATA_REPORT=$DATA_ROOT/report.json
DATA_REPORT_SHA256=$(sha256sum "$DATA_REPORT" | cut -d' ' -f1)
mkdir -p "$EVAL_ROOT" "$(dirname "$DISPATCH_RECEIPT")"
all_jobs=()
merge_jobs=()

for split in development holdout; do
  data=$DATA_ROOT/$split.jsonl
  data_sha256=$(sha256sum "$data" | cut -d' ' -f1)
  split_jobs=()
  shard_root=$EVAL_ROOT/$split/shards
  for shard in $(seq 0 $((SHARD_COUNT - 1))); do
    output=$shard_root/shard_${shard}
    exports="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
    exports+=",MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION"
    exports+=",MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256"
    exports+=",ADAPTER_CHECKPOINT=$ADAPTER_CHECKPOINT"
    exports+=",ADAPTER_CHECKPOINT_SHA256=$ADAPTER_CHECKPOINT_SHA256"
    exports+=",DATA=$data,DATA_SHA256=$data_sha256"
    exports+=",DATA_REPORT=$DATA_REPORT,DATA_REPORT_SHA256=$DATA_REPORT_SHA256"
    exports+=",SPLIT=$split,CANDIDATES_OUTPUT=$output/candidates.jsonl"
    exports+=",REPORT=$output/report.json,SHARD_INDEX=$shard,SHARD_COUNT=$SHARD_COUNT"
    job=$(sbatch --parsable --exclude="$EXCLUDE" --export="$exports" \
      "$RUNTIME/train/jobs/hf_idr1_evaluate_reviser.sbatch")
    split_jobs+=("$job")
    all_jobs+=("$job")
  done
  dependency=$(IFS=:; echo "${split_jobs[*]}")
  merged=$EVAL_ROOT/$split/merged
  merge_exports="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
  merge_exports+=",DATA=$data,DATA_SHA256=$data_sha256,SPLIT=$split"
  merge_exports+=",SHARD_ROOT=$shard_root,SHARD_COUNT=$SHARD_COUNT"
  merge_exports+=",CANDIDATES_OUTPUT=$merged/candidates.jsonl,REPORT=$merged/report.json"
  merge_job=$(sbatch --parsable --dependency=afterok:"$dependency" \
    --export="$merge_exports" "$RUNTIME/train/jobs/merge_idr_scale_evals.sbatch")
  merge_jobs+=("$merge_job")
done

temporary=$DISPATCH_RECEIPT.partial.$$
"$PYTHON" - "$temporary" "$SHARD_COUNT" "${all_jobs[*]}" "${merge_jobs[*]}" <<'PY'
import json
from pathlib import Path
import sys

path, shard_count, jobs, merges = sys.argv[1:]
Path(path).write_text(
    json.dumps(
        {
            "status": "submitted",
            "shard_count": int(shard_count),
            "evaluation_jobs": jobs.split(),
            "merge_jobs": merges.split(),
        },
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
mv "$temporary" "$DISPATCH_RECEIPT"
chmod a-w "$DISPATCH_RECEIPT"
printf 'evaluation_jobs=%s merge_jobs=%s\n' "${all_jobs[*]}" "${merge_jobs[*]}"
