#!/bin/bash
set -euo pipefail

: "${ALLOCATION_JOB_ID:?}" "${RUNTIME_ROOT:?}" "${ARTIFACT_ROOT:?}"
: "${BOARD_ROOT:?}" "${HOST:?}" "${MODEL_LOADER:?}"
: "${MODEL_ROOT:?}" "${MODEL_RECEIPT:?}" "${MODEL_REVISION:?}"
: "${MODEL_CONFIG_SHA256:?}" "${DRAFT_CHECKPOINT:?}"
: "${DRAFT_CHECKPOINT_SHA256:?}" "${REVISION_CHECKPOINT:?}"
: "${REVISION_CHECKPOINT_SHA256:?}" "${PYTHON:?}"

POLL_SECONDS=${POLL_SECONDS:-30}
BATCH_SIZE=${BATCH_SIZE:-16}
SCREEN_MANIFEST=${SCREEN_MANIFEST:-"$BOARD_ROOT/screen_manifest.json"}
ORDER=${ORDER:-"humaneval_plus mbpp_plus ifeval musr correctbench livebench livecodebench ruler longbench_pro mmlu_pro"}
CLAIM="$ARTIFACT_ROOT/generation_controller.json"

if [ "$POLL_SECONDS" -le 0 ] || [ "$BATCH_SIZE" -le 0 ]; then
  echo "poll interval and batch size must be positive" >&2
  exit 2
fi
case "$MODEL_LOADER" in
  causal|multimodal) ;;
  *) echo "model loader differs" >&2; exit 2 ;;
esac
for path in "$RUNTIME_ROOT" "$BOARD_ROOT" "$SCREEN_MANIFEST" \
  "$DRAFT_CHECKPOINT" "$REVISION_CHECKPOINT"; do
  [ -e "$path" ] && [ ! -L "$path" ] || {
    echo "required host-queue input differs: $path" >&2
    exit 2
  }
done

source_commit=$(git -C "$RUNTIME_ROOT" rev-parse HEAD)
if [ -n "$(git -C "$RUNTIME_ROOT" status --porcelain --untracked-files=all)" ]; then
  echo "host-queue runtime is dirty" >&2
  exit 2
fi

mkdir -p "$ARTIFACT_ROOT/logs"
if [ -e "$CLAIM" ] || [ -L "$CLAIM" ]; then
  echo "host-queue controller already claimed" >&2
  exit 2
fi
"$PYTHON" - "$CLAIM" "$source_commit" <<'PY'
import json
import os
import sys

path, source_commit = sys.argv[1:]
payload = {
    "schema": "shohin-dense-public-host-generation-controller-v1",
    "status": "waiting_for_allocation_and_model",
    "source_commit": source_commit,
    "duplicate_generation": False,
    "requeue": False,
}
data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(fd, "wb") as handle:
    handle.write(data)
    handle.flush()
    os.fsync(handle.fileno())
PY

while ! squeue -h -j "$ALLOCATION_JOB_ID" -t RUNNING | grep -q .; do
  state=$(sacct -X -n -j "$ALLOCATION_JOB_ID" --format=State -P | head -n 1)
  case "$state" in
    COMPLETED*|FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*)
      echo "allocation ended before host generation began: $state" >&2
      exit 2
      ;;
  esac
  sleep "$POLL_SECONDS"
done

while [ ! -s "$MODEL_RECEIPT" ] || [ ! -d "$MODEL_ROOT" ]; do
  if ! squeue -h -j "$ALLOCATION_JOB_ID" -t RUNNING | grep -q .; then
    echo "allocation ended before model restoration completed" >&2
    exit 2
  fi
  sleep "$POLL_SECONDS"
done

run_campaign() {
  local manifest=$1 output=$2 max_batch_tokens=$3
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
      --host "$HOST" \
      --manifest "$manifest" \
      --model-root "$MODEL_ROOT" \
      --model-source-root "$MODEL_ROOT" \
      --model-receipt "$MODEL_RECEIPT" \
      --model-revision "$MODEL_REVISION" \
      --model-config-sha256 "$MODEL_CONFIG_SHA256" \
      --model-loader "$MODEL_LOADER" \
      --draft-checkpoint "$DRAFT_CHECKPOINT" \
      --draft-checkpoint-sha256 "$DRAFT_CHECKPOINT_SHA256" \
      --revision-checkpoint "$REVISION_CHECKPOINT" \
      --revision-checkpoint-sha256 "$REVISION_CHECKPOINT_SHA256" \
      --output-root "$output" \
      --batch-size "$BATCH_SIZE" \
      --max-batch-tokens "$max_batch_tokens"
}

screen_output="$ARTIFACT_ROOT/screen_generation"
if [ ! -s "$screen_output/report.json" ]; then
  run_campaign "$SCREEN_MANIFEST" "$screen_output" 65536 \
    >"$ARTIFACT_ROOT/logs/screen.out" \
    2>"$ARTIFACT_ROOT/logs/screen.err"
fi
if [ ! -s "$screen_output/report.json" ]; then
  echo "screen campaign ended without a terminal report" >&2
  exit 2
fi

mkdir -p "$ARTIFACT_ROOT/full_generation"
for benchmark in $ORDER; do
  manifest="$BOARD_ROOT/manifests/$benchmark.json"
  output="$ARTIFACT_ROOT/full_generation/$benchmark"
  if [ ! -s "$manifest" ]; then
    echo "missing frozen shared-board manifest: $manifest" >&2
    exit 2
  fi
  if [ -s "$output/report.json" ]; then
    continue
  fi
  max_batch_tokens=65536
  if [ "$benchmark" = ruler ]; then
    max_batch_tokens=131072
  elif [ "$benchmark" = longbench_pro ]; then
    max_batch_tokens=196608
  fi
  run_campaign "$manifest" "$output" "$max_batch_tokens" \
    >"$ARTIFACT_ROOT/logs/full_${benchmark}.out" \
    2>"$ARTIFACT_ROOT/logs/full_${benchmark}.err"
  if [ ! -s "$output/report.json" ]; then
    echo "$benchmark ended without a terminal campaign report" >&2
    exit 2
  fi
done

"$PYTHON" - "$ARTIFACT_ROOT/generation_complete.json" "$source_commit" <<'PY'
import json
import os
import sys

path, source_commit = sys.argv[1:]
payload = {
    "schema": "shohin-dense-public-host-generation-complete-v1",
    "status": "complete",
    "source_commit": source_commit,
    "benchmarks": 10,
    "duplicate_generation": False,
    "requeue": False,
}
data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(fd, "wb") as handle:
    handle.write(data)
    handle.flush()
    os.fsync(handle.fileno())
PY
