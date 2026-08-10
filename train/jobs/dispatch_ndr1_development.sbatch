#!/bin/bash
#SBATCH --job-name=ndr1-eval-dispatch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/ndr1_eval_dispatch_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/ndr1_eval_dispatch_%j.err

set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
SHARD_COUNT=${SHARD_COUNT:-4}
EXCLUDE_NODES=${EXCLUDE_NODES:-evc26:evc31:evc33:evc38:evc46}
EXCLUDE=${EXCLUDE_NODES//:/,}

for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION \
  MODEL_CONFIG_SHA256 MODEL_MANIFEST_SHA256 DEVELOPMENT DEVELOPMENT_SHA256 \
  DEVELOPMENT_REPORT DEVELOPMENT_REPORT_SHA256 NDR_DATA_REPORT \
  NDR_DATA_REPORT_SHA256 ALIGNED_TRAIN_ROOT SHUFFLED_TRAIN_ROOT OUTPUT_ROOT \
  RECEIPT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done

test "$SHARD_COUNT" -ge 2
test -x "$PYTHON"
test ! -e "$OUTPUT_ROOT"
test ! -e "$RECEIPT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = \
  "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = \
  "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$MODEL_ROOT/SHA256SUMS" | cut -d' ' -f1)" = \
  "$MODEL_MANIFEST_SHA256"
test "$(sha256sum "$DEVELOPMENT" | cut -d' ' -f1)" = "$DEVELOPMENT_SHA256"
test "$(sha256sum "$DEVELOPMENT_REPORT" | cut -d' ' -f1)" = \
  "$DEVELOPMENT_REPORT_SHA256"
test "$(sha256sum "$NDR_DATA_REPORT" | cut -d' ' -f1)" = \
  "$NDR_DATA_REPORT_SHA256"

mkdir -p "$OUTPUT_ROOT" "$(dirname "$RECEIPT")"
declare -A checkpoint_hashes fit_reports merge_jobs
all_eval_jobs=()

for arm in aligned shuffled; do
  if [[ "$arm" == aligned ]]; then
    train_root=$ALIGNED_TRAIN_ROOT
  else
    train_root=$SHUFFLED_TRAIN_ROOT
  fi
  checkpoint=$train_root/checkpoint_0000512.pt
  fit_report=$train_root/report.json
  test -f "$checkpoint"
  test -f "$fit_report"
  checkpoint_hashes[$arm]=$(sha256sum "$checkpoint" | cut -d' ' -f1)
  fit_reports[$arm]=$fit_report

  eval_root=$OUTPUT_ROOT/$arm
  shard_jobs=()
  shard_reports=()
  for shard in $(seq 0 $((SHARD_COUNT - 1))); do
    shard_root=$eval_root/shards/shard_$shard
    exports="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
    exports+=",MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION"
    exports+=",MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256"
    exports+=",MODEL_MANIFEST_SHA256=$MODEL_MANIFEST_SHA256,MODEL_LOADER=multimodal"
    exports+=",ADAPTER_CHECKPOINT=$checkpoint"
    exports+=",ADAPTER_CHECKPOINT_SHA256=${checkpoint_hashes[$arm]}"
    exports+=",DATA=$DEVELOPMENT,DATA_SHA256=$DEVELOPMENT_SHA256"
    exports+=",DATA_REPORT=$DEVELOPMENT_REPORT"
    exports+=",DATA_REPORT_SHA256=$DEVELOPMENT_REPORT_SHA256,SPLIT=development"
    exports+=",SHARD_INDEX=$shard,SHARD_COUNT=$SHARD_COUNT,STAGE_MODEL=true"
    exports+=",CANDIDATES_OUTPUT=$shard_root/candidates.jsonl"
    exports+=",REPORT=$shard_root/report.json"
    job=$(sbatch --parsable --exclude="$EXCLUDE" --export="$exports" \
      "$RUNTIME/train/jobs/hf_idr1_evaluate_reviser.sbatch")
    shard_jobs+=("$job")
    shard_reports+=("$shard_root/report.json")
    all_eval_jobs+=("$job")
  done

  dependency=$(IFS=:; echo "${shard_jobs[*]}")
  merge_args=""
  for shard_report in "${shard_reports[@]}"; do
    merge_args+=" --shard-report $shard_report"
  done
  merge_jobs[$arm]=$(sbatch --parsable --dependency=afterok:"$dependency" \
    --partition=normal --time=00:20:00 --cpus-per-task=1 --mem=8G \
    --wrap="set -euo pipefail; export PYTHONPATH=$RUNTIME/train:$RUNTIME/pipeline; $PYTHON $RUNTIME/pipeline/merge_idr1_evaluation_shards.py --data $DEVELOPMENT --split development $merge_args --candidates-output $eval_root/candidates.jsonl --report $eval_root/report.json; chmod a-w $eval_root/candidates.jsonl $eval_root/report.json")
done

dependency=afterok:${merge_jobs[aligned]}:${merge_jobs[shuffled]}
comparison_job=$(sbatch --parsable --dependency="$dependency" \
  --partition=normal --time=00:10:00 --cpus-per-task=1 --mem=2G \
  --wrap="set -euo pipefail; export PYTHONPATH=$RUNTIME/train:$RUNTIME/pipeline; $PYTHON $RUNTIME/pipeline/compare_ndr1_development.py --aligned-report $OUTPUT_ROOT/aligned/report.json --shuffled-report $OUTPUT_ROOT/shuffled/report.json --aligned-fit ${fit_reports[aligned]} --shuffled-fit ${fit_reports[shuffled]} --data-report $NDR_DATA_REPORT --output $OUTPUT_ROOT/comparison.json; chmod a-w $OUTPUT_ROOT/comparison.json")

temporary=$RECEIPT.partial.$$
"$PYTHON" - "$temporary" "$SHARD_COUNT" "${all_eval_jobs[*]}" \
  "${merge_jobs[aligned]}" "${merge_jobs[shuffled]}" "$comparison_job" \
  "${checkpoint_hashes[aligned]}" "${checkpoint_hashes[shuffled]}" <<'PY'
import json
from pathlib import Path
import sys

(
    path,
    shard_count,
    evaluation_jobs,
    aligned_merge,
    shuffled_merge,
    comparison_job,
    aligned_checkpoint_sha256,
    shuffled_checkpoint_sha256,
) = sys.argv[1:]
Path(path).write_text(
    json.dumps(
        {
            "aligned_checkpoint_sha256": aligned_checkpoint_sha256,
            "aligned_merge_job": aligned_merge,
            "comparison_job": comparison_job,
            "evaluation_jobs": evaluation_jobs.split(),
            "holdout_opened": False,
            "shard_count_per_arm": int(shard_count),
            "shuffled_checkpoint_sha256": shuffled_checkpoint_sha256,
            "shuffled_merge_job": shuffled_merge,
            "status": "submitted",
        },
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
mv "$temporary" "$RECEIPT"
chmod a-w "$RECEIPT"
cat "$RECEIPT"
