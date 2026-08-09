#!/bin/bash
#SBATCH --job-name=kr2-dispatch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/kr2_dispatch_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/kr2_dispatch_%j.err

set -euo pipefail

EXCLUDE_NODES=${EXCLUDE_NODES:-evc33,evc38}

for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION \
  MODEL_CONFIG_SHA256 MODEL_MANIFEST_SHA256 WARM_START WARM_START_SHA256 \
  DATA_ROOT OUTPUT_ROOT DEPTH_ONE_ROOT RECEIPT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
test ! -e "$OUTPUT_ROOT"
test ! -e "$RECEIPT"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = \
  "$RUNTIME_MANIFEST_SHA256"
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = \
  "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$MODEL_ROOT/SHA256SUMS" | cut -d' ' -f1)" = \
  "$MODEL_MANIFEST_SHA256"
test "$(sha256sum "$WARM_START" | cut -d' ' -f1)" = "$WARM_START_SHA256"
mkdir -p "$OUTPUT_ROOT" "$(dirname "$RECEIPT")"

declare -A train_jobs merge_jobs rescore_jobs
for arm in keep_or_repair direct_rewrite; do
  data=$DATA_ROOT/$arm/train.jsonl
  data_report=$DATA_ROOT/$arm/report.json
  data_sha=$(sha256sum "$data" | cut -d' ' -f1)
  data_report_sha=$(sha256sum "$data_report" | cut -d' ' -f1)
  train_output=$OUTPUT_ROOT/$arm/train
  train_exports="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
  train_exports+=",MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION"
  train_exports+=",MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256,MODEL_LOADER=multimodal"
  train_exports+=",WARM_START=$WARM_START,WARM_START_SHA256=$WARM_START_SHA256"
  train_exports+=",DATA=$data,DATA_SHA256=$data_sha"
  train_exports+=",DATA_REPORT=$data_report,DATA_REPORT_SHA256=$data_report_sha"
  train_exports+=",OUTPUT=$train_output"
  train_jobs[$arm]=$(sbatch --parsable --exclude="$EXCLUDE_NODES" \
    --export="$train_exports" "$RUNTIME/train/jobs/hf_idr1_train_reviser.sbatch")

  eval_root=$OUTPUT_ROOT/$arm/development
  development=$DATA_ROOT/$arm/development.jsonl
  development_sha=$(sha256sum "$development" | cut -d' ' -f1)
  shard_jobs=()
  shard_reports=()
  for shard in $(seq 0 7); do
    shard_root=$eval_root/shard_$shard
    mkdir -p "$shard_root"
    eval_exports="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
    eval_exports+=",MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION"
    eval_exports+=",MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256"
    eval_exports+=",MODEL_MANIFEST_SHA256=$MODEL_MANIFEST_SHA256,MODEL_LOADER=multimodal"
    eval_exports+=",TRAIN_OUTPUT=$train_output,WARM_START_SHA256=$WARM_START_SHA256"
    eval_exports+=",DATA=$development"
    eval_exports+=",DATA_SHA256=$development_sha,DATA_REPORT=$data_report"
    eval_exports+=",DATA_REPORT_SHA256=$data_report_sha,SPLIT=development"
    eval_exports+=",SHARD_INDEX=$shard,SHARD_COUNT=8,STAGE_MODEL=true"
    eval_exports+=",CANDIDATES_OUTPUT=$shard_root/candidates.jsonl"
    eval_exports+=",REPORT=$shard_root/report.json"
    shard_job=$(sbatch --parsable --exclude="$EXCLUDE_NODES" \
      --dependency=afterok:${train_jobs[$arm]} --export="$eval_exports" \
      "$RUNTIME/train/jobs/hf_kr2_evaluate.sbatch")
    shard_jobs+=("$shard_job")
    shard_reports+=("$shard_root/report.json")
  done
  dependency=$(IFS=:; echo "${shard_jobs[*]}")
  merge_args=""
  for shard_report in "${shard_reports[@]}"; do
    merge_args+=" --shard-report $shard_report"
  done
  merge_jobs[$arm]=$(sbatch --parsable --dependency=afterok:$dependency \
    --partition=normal --time=00:20:00 --cpus-per-task=1 --mem=8G \
    --wrap="set -euo pipefail; export PYTHONPATH=$RUNTIME/train:$RUNTIME/pipeline; /lustre/fs1/home/sa305415/shohin/envs/product-reasoning-b3a3603-r2/bin/python $RUNTIME/pipeline/merge_idr1_evaluation_shards.py --data $development --split development $merge_args --candidates-output $eval_root/raw_candidates.jsonl --report $eval_root/raw_report.json; chmod a-w $eval_root/raw_candidates.jsonl $eval_root/raw_report.json")
  selected=$eval_root/selected_candidates.jsonl
  selected_report=$eval_root/selected_report.json
  rescore_jobs[$arm]=$(sbatch --parsable --dependency=afterok:${merge_jobs[$arm]} \
    --partition=normal --time=00:30:00 --cpus-per-task=1 --mem=4G \
    --wrap="set -euo pipefail; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$RUNTIME/train:$RUNTIME/pipeline; /lustre/fs1/home/sa305415/shohin/envs/product-reasoning-b3a3603-r2/bin/python $RUNTIME/pipeline/rescore_kr2_keep_actions.py --mode $arm --data $development --raw-candidates $eval_root/raw_candidates.jsonl --raw-report $eval_root/raw_report.json --output-candidates $selected --output-report $selected_report; chmod a-w $selected $selected_report")
done

compare_dependency=afterok:${rescore_jobs[keep_or_repair]}:${rescore_jobs[direct_rewrite]}
comparison_job=$(sbatch --parsable --dependency=$compare_dependency \
  --partition=normal --time=00:10:00 --cpus-per-task=1 --mem=2G \
  --wrap="set -euo pipefail; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$RUNTIME/train:$RUNTIME/pipeline; /lustre/fs1/home/sa305415/shohin/envs/product-reasoning-b3a3603-r2/bin/python $RUNTIME/pipeline/compare_kr2_stage_owner.py --treatment-candidates $OUTPUT_ROOT/keep_or_repair/development/selected_candidates.jsonl --treatment-report $OUTPUT_ROOT/keep_or_repair/development/selected_report.json --direct-candidates $OUTPUT_ROOT/direct_rewrite/development/selected_candidates.jsonl --direct-report $OUTPUT_ROOT/direct_rewrite/development/selected_report.json --depth-one-candidates $DEPTH_ONE_ROOT/candidates.jsonl --output $OUTPUT_ROOT/comparison.json; chmod a-w $OUTPUT_ROOT/comparison.json")

temporary=$RECEIPT.partial.$$
cat >"$temporary" <<EOF
{"comparison_job":"$comparison_job","direct_merge_job":"${merge_jobs[direct_rewrite]}","direct_rescore_job":"${rescore_jobs[direct_rewrite]}","direct_train_job":"${train_jobs[direct_rewrite]}","exclude_nodes":"$EXCLUDE_NODES","keep_merge_job":"${merge_jobs[keep_or_repair]}","keep_rescore_job":"${rescore_jobs[keep_or_repair]}","keep_train_job":"${train_jobs[keep_or_repair]}","runtime_manifest_sha256":"$RUNTIME_MANIFEST_SHA256","status":"submitted"}
EOF
mv "$temporary" "$RECEIPT"
chmod a-w "$RECEIPT"
cat "$RECEIPT"
