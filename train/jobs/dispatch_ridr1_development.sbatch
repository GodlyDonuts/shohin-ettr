#!/bin/bash
#SBATCH --job-name=ridr1-dispatch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/ridr1_dispatch_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/ridr1_dispatch_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${RUNTIME_MANIFEST_SHA256:?RUNTIME_MANIFEST_SHA256 is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${MODEL_REVISION:?MODEL_REVISION is required}"
: "${MODEL_CONFIG_SHA256:?MODEL_CONFIG_SHA256 is required}"
: "${MODEL_MANIFEST_SHA256:?MODEL_MANIFEST_SHA256 is required}"
: "${ADAPTER_CHECKPOINT:?ADAPTER_CHECKPOINT is required}"
: "${ADAPTER_CHECKPOINT_SHA256:?ADAPTER_CHECKPOINT_SHA256 is required}"
: "${DATA_ROOT:?DATA_ROOT is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${DEPTH_ONE_ROOT:?DEPTH_ONE_ROOT is required}"

DATA=$DATA_ROOT/development.jsonl
DATA_REPORT=$DATA_ROOT/report.json
DATA_SHA256=$(sha256sum "$DATA" | cut -d' ' -f1)
DATA_REPORT_SHA256=$(sha256sum "$DATA_REPORT" | cut -d' ' -f1)
mkdir -p "$OUTPUT_ROOT"
test ! -e "$OUTPUT_ROOT/comparison.json"

common="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION,MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256,MODEL_MANIFEST_SHA256=$MODEL_MANIFEST_SHA256,MODEL_LOADER=multimodal,ADAPTER_CHECKPOINT=$ADAPTER_CHECKPOINT,ADAPTER_CHECKPOINT_SHA256=$ADAPTER_CHECKPOINT_SHA256,DATA=$DATA,DATA_SHA256=$DATA_SHA256,DATA_REPORT=$DATA_REPORT,DATA_REPORT_SHA256=$DATA_REPORT_SHA256,SPLIT=development,SHARD_COUNT=8,STAGE_MODEL=true"
jobs=()
reports=()
for shard in $(seq 0 7); do
  root=$OUTPUT_ROOT/shard_$shard
  mkdir -p "$root"
  job=$(sbatch --parsable --export="$common,SHARD_INDEX=$shard,CANDIDATES_OUTPUT=$root/candidates.jsonl,REPORT=$root/report.json" "$RUNTIME/train/jobs/hf_idr1_evaluate_reviser.sbatch")
  jobs+=("$job")
  reports+=("$root/report.json")
done
dependency=$(IFS=:; echo "${jobs[*]}")
merge_args=""
for report in "${reports[@]}"; do
  merge_args+=" --shard-report $report"
done
merge=$(sbatch --parsable --dependency="afterok:$dependency" --partition=normal --time=00:20:00 --cpus-per-task=1 --mem=8G --wrap="set -euo pipefail; export PYTHONPATH=$RUNTIME/train:$RUNTIME/pipeline; /lustre/fs1/home/sa305415/shohin/envs/product-reasoning-b3a3603-r2/bin/python $RUNTIME/pipeline/merge_idr1_evaluation_shards.py --data $DATA --split development $merge_args --candidates-output $OUTPUT_ROOT/candidates.jsonl --report $OUTPUT_ROOT/report.json; chmod a-w $OUTPUT_ROOT/candidates.jsonl $OUTPUT_ROOT/report.json")
compare=$(sbatch --parsable --dependency="afterok:$merge" --partition=normal --time=00:10:00 --cpus-per-task=1 --mem=2G --wrap="set -euo pipefail; export PYTHONPATH=$RUNTIME/train:$RUNTIME/pipeline; $RUNTIME/pipeline/compare_recursive_idr.py --depth-one-candidates $DEPTH_ONE_ROOT/candidates.jsonl --depth-one-report $DEPTH_ONE_ROOT/report.json --depth-two-candidates $OUTPUT_ROOT/candidates.jsonl --depth-two-report $OUTPUT_ROOT/report.json --output $OUTPUT_ROOT/comparison.json; chmod a-w $OUTPUT_ROOT/comparison.json")
printf 'shards=%s\nmerge=%s\ncompare=%s\n' "${jobs[*]}" "$merge" "$compare"
