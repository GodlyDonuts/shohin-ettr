#!/bin/bash
set -euo pipefail

: "${ALLOCATION_JOB_ID:?}" "${RUNTIME_ROOT:?}" "${ARTIFACT_ROOT:?}"
: "${MODEL_ROOT:?}" "${MODEL_RECEIPT:?}" "${MODEL_REVISION:?}"
: "${MODEL_CONFIG_SHA256:?}" "${DRAFT_CHECKPOINT:?}" "${DRAFT_CHECKPOINT_SHA256:?}"
: "${REVISION_CHECKPOINT:?}" "${REVISION_CHECKPOINT_SHA256:?}" "${PYTHON:?}"

SCREEN_REPORT=${SCREEN_REPORT:-"$ARTIFACT_ROOT/screen_generation/report.json"}
POLL_SECONDS=${POLL_SECONDS:-30}
BATCH_SIZE=${BATCH_SIZE:-16}
ORDER=${ORDER:-"humaneval_plus mbpp_plus ifeval musr correctbench livebench livecodebench ruler longbench_pro mmlu_pro"}

if [ "$POLL_SECONDS" -le 0 ] || [ "$BATCH_SIZE" -le 0 ]; then
  echo "poll interval and batch size must be positive" >&2
  exit 2
fi

while [ ! -s "$SCREEN_REPORT" ]; do
  if ! squeue -h -j "$ALLOCATION_JOB_ID" -t RUNNING | grep -q .; then
    echo "allocation ended before the screen campaign completed" >&2
    exit 2
  fi
  sleep "$POLL_SECONDS"
done

mkdir -p "$ARTIFACT_ROOT/full_generation" "$ARTIFACT_ROOT/logs"
for benchmark in $ORDER; do
  manifest="$ARTIFACT_ROOT/manifests/$benchmark.json"
  output="$ARTIFACT_ROOT/full_generation/$benchmark"
  if [ ! -s "$manifest" ]; then
    echo "missing frozen manifest: $manifest" >&2
    exit 2
  fi
  if [ -s "$output/report.json" ]; then
    continue
  fi
  max_batch_tokens=65536
  if [ "$benchmark" = "ruler" ]; then
    max_batch_tokens=131072
  elif [ "$benchmark" = "longbench_pro" ]; then
    max_batch_tokens=196608
  fi
  srun \
    --jobid="$ALLOCATION_JOB_ID" \
    --overlap \
    --ntasks=1 \
    --cpus-per-task=16 \
    --mem=120G \
    env \
      PYTHONDONTWRITEBYTECODE=1 \
      HF_HUB_OFFLINE=1 \
      TRANSFORMERS_OFFLINE=1 \
      TOKENIZERS_PARALLELISM=false \
      PYTHONPATH="$RUNTIME_ROOT/pipeline:$RUNTIME_ROOT/train" \
    "$PYTHON" "$RUNTIME_ROOT/train/hf_dense_public_benchmark_campaign.py" \
      --host qwen3.5-9b \
      --manifest "$manifest" \
      --model-root "$MODEL_ROOT" \
      --model-source-root "$MODEL_ROOT" \
      --model-receipt "$MODEL_RECEIPT" \
      --model-revision "$MODEL_REVISION" \
      --model-config-sha256 "$MODEL_CONFIG_SHA256" \
      --model-loader multimodal \
      --draft-checkpoint "$DRAFT_CHECKPOINT" \
      --draft-checkpoint-sha256 "$DRAFT_CHECKPOINT_SHA256" \
      --revision-checkpoint "$REVISION_CHECKPOINT" \
      --revision-checkpoint-sha256 "$REVISION_CHECKPOINT_SHA256" \
      --output-root "$output" \
      --batch-size "$BATCH_SIZE" \
      --max-batch-tokens "$max_batch_tokens" \
      >"$ARTIFACT_ROOT/logs/full_${benchmark}.out" \
      2>"$ARTIFACT_ROOT/logs/full_${benchmark}.err"
  if [ ! -s "$output/report.json" ]; then
    echo "$benchmark ended without a terminal campaign report" >&2
    exit 2
  fi
done
