#!/bin/bash
set -euo pipefail

REPOSITORY_ROOT=${REPOSITORY_ROOT:?}
PYTHON=${PYTHON:-/lustre/fs1/home/sa305415/shohin/envs/product-reasoning-b3a3603-r2/bin/python}
RUN_ROOT=${RUN_ROOT:?}
SUBMIT=${SUBMIT:-0}
SHARDS=8
EXCLUDES=evc26,evc29,evc31,evc32,evc38,evc46
PATH_EXPORT=/apps/slurm/current/bin:/usr/bin:/bin

BASE_PARENT=/lustre/fs1/home/sa305415/shohin/artifacts/dense_public_bases
SOURCE_ROOT="$RUN_ROOT/sources"
DATA_ROOT="$RUN_ROOT/data"
TRAIN_MATH=/lustre/fs1/home/sa305415/shohin/artifacts/product_reasoning/router_outcomes/cvg1_disjoint_5e63a06/math_disjoint4096_2026080709_r2.jsonl
TRAIN_SCIENCE=/lustre/fs1/home/sa305415/shohin/artifacts/product_reasoning/router_outcomes/cvg1_disjoint_5e63a06/science_disjoint4096_2026080709_r2.jsonl
TRAIN_CODE=/lustre/fs1/home/sa305415/shohin/artifacts/product_reasoning/router_outcomes/cvg1_code_disjoint_1b49ca7/mbpp_disjoint200_2026080715_r1.jsonl

if [ "$SUBMIT" != 1 ]; then
  printf '%s\n' \
    'dense-public dry run: 3 hosts, 3 benchmarks/host, 8 independent one-H100 shards/benchmark' \
    'GPU requests: 75 exact single-H100 jobs (3 mechanics + 72 benchmark shards)' \
    'CPU jobs: 19; bases restore/reclaim sequentially qwen9 -> smollm3 -> olmo2'
  exit 0
fi

test ! -e "$RUN_ROOT"
for path in "$REPOSITORY_ROOT" "$PYTHON" "$TRAIN_MATH" "$TRAIN_SCIENCE" "$TRAIN_CODE"; do
  test -e "$path"
done
mkdir -m 700 -p "$RUN_ROOT/logs" "$BASE_PARENT"
jobs_tsv="$RUN_ROOT/jobs.tsv"
: > "$jobs_tsv"

submit_job() {
  local name=$1 dependency=$2 script=$3 exports=$4
  local args=(--parsable --job-name "$name" --nice=10000 --output "$RUN_ROOT/logs/%x-%A_%a.out" --error "$RUN_ROOT/logs/%x-%A_%a.err" --export "PATH=$PATH_EXPORT,$exports")
  if [ -n "$dependency" ]; then
    args+=(--dependency "afterok:$dependency")
  fi
  if grep -q 'gres=gpu' "$script"; then
    args+=(--exclude "$EXCLUDES")
  fi
  local job
  job=$(sbatch "${args[@]}" "$script")
  job=${job%%;*}
  printf '%s\t%s\n' "$name" "$job" >> "$jobs_tsv"
  printf '%s' "$job"
}

common="REPOSITORY_ROOT=$REPOSITORY_ROOT,PYTHON=$PYTHON"
data_exports="$common,SOURCE_ROOT=$SOURCE_ROOT,DATA_ROOT=$DATA_ROOT,TRAIN_MATH=$TRAIN_MATH,TRAIN_SCIENCE=$TRAIN_SCIENCE,TRAIN_CODE=$TRAIN_CODE"
data_job=$(submit_job dense-public-data '' "$REPOSITORY_ROOT/pipeline/jobs/dense_public_prepare_data.sbatch" "$data_exports")

previous_reclaim=''
for host in qwen9 smollm3 olmo2; do
  case "$host" in
    qwen9)
      repository=Qwen/Qwen3.5-9B
      revision=c202236235762e1c871ad0ccb60c8ee5ba337b9a
      config=d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05
      loader=multimodal
      model_max=26843545600
      draft=/lustre/fs1/home/sa305415/shohin/artifacts/product_reasoning/idr_aqc_release_8f0bd8d_r1/draft_adapter.pt
      draft_sha=854a7cc44fbc2b54418f4e5bd09b7efeed0da44fc9ce217b0bb6b1997b722971
      treatment=/lustre/fs1/home/sa305415/shohin/artifacts/product_reasoning/idr_aqc_release_8f0bd8d_r1/revision_adapter.pt
      treatment_sha=df3c264d426941fef8ba9c10a90fe9fab304ec2864738209a4d79f9f81e0c473
      release=/lustre/fs1/home/sa305415/shohin/artifacts/product_reasoning/idr_aqc_release_8f0bd8d_r1
      draft_base=0
      ;;
    smollm3)
      repository=HuggingFaceTB/SmolLM3-3B
      revision=a07cc9a04f16550a088caea529712d1d335b0ac1
      config=c72b1031274ff4626e434d0019e88e95a767460135db9ee492eb80652b786af1
      loader=causal
      model_max=10737418240
      draft=/lustre/fs1/home/sa305415/shohin/artifacts/product_reasoning/ttr1_smollm3_3b_ddba463_r1/checkpoint_0001000.normalized.pt
      draft_sha=b260d1acb20931e53f9f380f67a9d6b3feab89ae26f79dabb874f991f9c10edb
      treatment=/lustre/fs1/home/sa305415/shohin/artifacts/product_reasoning/ttr1_smollm3_3b_5f4a83b_r6/treatment_fit/checkpoint_0000256.pt
      treatment_sha=e2b7a1798aa9430e139118222d3e469de42dc8cfd9affc954819ab5b0db37691
      release=''
      draft_base=0
      ;;
    olmo2)
      repository=allenai/OLMo-2-1124-7B-Instruct
      revision=470b1fba1ae01581f270116362ee4aa1b97f4c84
      config=ff8cc8709a229515676797ab6f343a09391041c9a8fbbc78bfec5be4c2e3664e
      loader=causal
      model_max=21474836480
      draft=''
      draft_sha=''
      treatment=/lustre/fs1/home/sa305415/shohin/artifacts/product_reasoning/sctr1_olmo2_7b_bbe3a7a_r2/campaign_r1/always_revise_fit/checkpoint_0000256.pt
      treatment_sha=24105cd0ad524cadfb50ba1c78a94da56eb0fdcbfb28df03501786ebbc3da6a1
      release=''
      draft_base=1
      ;;
  esac
  model_root="$BASE_PARENT/${host}_${revision}_r1"
  model_receipt="$BASE_PARENT/${host}_${revision}_r1.receipt.json"
  restore_dependency=$previous_reclaim
  restore_exports="$common,MODEL_REPOSITORY=$repository,MODEL_REVISION=$revision,MODEL_CONFIG_SHA256=$config,MODEL_ROOT=$model_root,MODEL_RECEIPT=$model_receipt,MODEL_EXPECTED_BYTES_MAX=$model_max"
  restore_job=$(submit_job "dense-$host-restore" "$restore_dependency" "$REPOSITORY_ROOT/train/jobs/dense_public_restore.sbatch" "$restore_exports")
  base_generation="$common,HOST=$host,MODEL_ROOT=$model_root,MODEL_RECEIPT=$model_receipt,MODEL_REVISION=$revision,MODEL_CONFIG_SHA256=$config,MODEL_LOADER=$loader,REVISION_CHECKPOINT=$treatment,REVISION_CHECKPOINT_SHA256=$treatment_sha,DRAFT_BASE=$draft_base,DRAFT_CHECKPOINT=$draft,DRAFT_CHECKPOINT_SHA256=$draft_sha,RELEASE_ROOT=$release,BATCH_SIZE=1,BATCH_PAIRS=2"
  mechanics_report="$RUN_ROOT/$host/mechanics/report.json"
  mechanics_exports="$base_generation,BENCHMARK=mmlu_pro,QUESTIONS=$DATA_ROOT/smoke.questions.jsonl,REPORT=$mechanics_report,SHARD_INDEX=0,SHARD_COUNT=1,MAX_NEW_TOKENS=128"
  mechanics_job=$(submit_job "dense-$host-mechanics" "$data_job:$restore_job" "$REPOSITORY_ROOT/train/jobs/dense_public_generate.sbatch" "$mechanics_exports")
  score_jobs=()
  for benchmark in mmlu_pro ifeval musr; do
    case "$benchmark" in
      mmlu_pro|ifeval) max_tokens=2048 ;;
      musr) max_tokens=2400 ;;
    esac
    shard_jobs=()
    index=0
    while [ "$index" -lt "$SHARDS" ]; do
      report="$RUN_ROOT/$host/generation/$benchmark/shard_${index}/report.json"
      exports="$base_generation,BENCHMARK=$benchmark,QUESTIONS=$DATA_ROOT/$benchmark/screen.questions.jsonl,REPORT=$report,SHARD_INDEX=$index,SHARD_COUNT=$SHARDS,MAX_NEW_TOKENS=$max_tokens"
      shard_jobs+=("$(submit_job "dense-$host-$benchmark-s$index" "$mechanics_job" "$REPOSITORY_ROOT/train/jobs/dense_public_generate.sbatch" "$exports")")
      index=$((index + 1))
    done
    dependency=$(IFS=:; echo "${shard_jobs[*]}")
    score_output="$RUN_ROOT/$host/scores/$benchmark.json"
    score_exports="$common,BENCHMARK=$benchmark,GENERATION_ROOT=$RUN_ROOT/$host/generation/$benchmark,ASSESSORS=$DATA_ROOT/$benchmark/screen.assessors.jsonl,IFEVAL_ROOT=$SOURCE_ROOT/google-research/instruction_following_eval,OUTPUT=$score_output,SHARD_COUNT=$SHARDS"
    score_jobs+=("$(submit_job "dense-$host-$benchmark-score" "$dependency" "$REPOSITORY_ROOT/pipeline/jobs/dense_public_score.sbatch" "$score_exports")")
  done
  score_dependency=$(IFS=:; echo "${score_jobs[*]}")
  analysis_output="$RUN_ROOT/$host/analysis.json"
  analysis_exports="$common,SCORE_ROOT=$RUN_ROOT/$host/scores,OUTPUT=$analysis_output"
  analysis_job=$(submit_job "dense-$host-analyze" "$score_dependency" "$REPOSITORY_ROOT/pipeline/jobs/dense_public_analyze.sbatch" "$analysis_exports")
  reclaim_output="$RUN_ROOT/$host/model_reclaim.json"
  reclaim_exports="$common,MODEL_ROOT=$model_root,MODEL_RECEIPT=$model_receipt,OUTPUT=$reclaim_output"
  previous_reclaim=$(submit_job "dense-$host-reclaim" "$analysis_job" "$REPOSITORY_ROOT/pipeline/jobs/dense_public_reclaim_model.sbatch" "$reclaim_exports")
done

"$PYTHON" - "$jobs_tsv" "$RUN_ROOT/dispatch.json" <<'PY'
import json
from pathlib import Path
import sys

rows = []
for line in Path(sys.argv[1]).read_text().splitlines():
    name, job_id = line.split("\t")
    rows.append({"name": name, "job_id": int(job_id)})
payload = {
    "schema": "shohin-dense-public-dispatch-v1",
    "status": "submitted",
    "jobs": rows,
    "job_count": len(rows),
    "single_h100_jobs": 75,
    "hosts": ["qwen9", "smollm3", "olmo2"],
    "benchmarks": ["mmlu_pro", "ifeval", "musr"],
    "shards_per_benchmark": 8,
    "nice": 10000,
    "requeue": False,
}
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
chmod 444 "$jobs_tsv" "$RUN_ROOT/dispatch.json"
printf 'submitted %s jobs; terminal reclaim job %s\n' "$(wc -l < "$jobs_tsv")" "$previous_reclaim"
