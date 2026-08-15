#!/bin/bash
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${REPOSITORY_ROOT:?}" "${PYTHON:?}" "${HOST:?}" "${BENCHMARK:?}"
: "${MODEL_ROOT:?}" "${MODEL_RECEIPT:?}" "${MODEL_REVISION:?}"
: "${MODEL_CONFIG_SHA256:?}" "${MODEL_LOADER:?}" "${QUESTIONS:?}"
: "${REVISION_CHECKPOINT:?}" "${REVISION_CHECKPOINT_SHA256:?}"
: "${REPORT:?}" "${SHARD_INDEX:?}" "${SHARD_COUNT:?}" "${MAX_NEW_TOKENS:?}"

work="/tmp/dense-public-${SLURM_JOB_ID}"
test ! -e "$work"
mkdir -m 700 "$work"
trap 'rm -rf -- "$work"' EXIT
model_stage="$work/model"
mkdir "$model_stage"
cp -a "$MODEL_ROOT/." "$model_stage/"

export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$REPOSITORY_ROOT/pipeline:$REPOSITORY_ROOT/train"

"$PYTHON" - "$MODEL_RECEIPT" "$model_stage" <<'PY'
import json
from pathlib import Path
import sys
from restore_dense_public_model import tree_manifest

receipt = json.loads(Path(sys.argv[1]).read_text())
rows, digest, total = tree_manifest(Path(sys.argv[2]))
if digest != receipt["tree_sha256"] or len(rows) != receipt["files"] or total != receipt["bytes"]:
    raise SystemExit("staged model tree differs")
PY

arguments=(
  --host "$HOST"
  --benchmark "$BENCHMARK"
  --model-root "$model_stage"
  --model-source-root "$MODEL_ROOT"
  --model-receipt "$MODEL_RECEIPT"
  --model-revision "$MODEL_REVISION"
  --model-config-sha256 "$MODEL_CONFIG_SHA256"
  --model-loader "$MODEL_LOADER"
  --revision-checkpoint "$REVISION_CHECKPOINT"
  --revision-checkpoint-sha256 "$REVISION_CHECKPOINT_SHA256"
  --questions "$QUESTIONS"
  --report "$REPORT"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --batch-size "${BATCH_SIZE:-1}"
  --batch-pairs "${BATCH_PAIRS:-2}"
  --seed 2026081519
  --shard-index "$SHARD_INDEX"
  --shard-count "$SHARD_COUNT"
)
if [ "${DRAFT_BASE:-0}" = 1 ]; then
  arguments+=(--draft-base)
else
  : "${DRAFT_CHECKPOINT:?}" "${DRAFT_CHECKPOINT_SHA256:?}"
  arguments+=(--draft-checkpoint "$DRAFT_CHECKPOINT")
  arguments+=(--draft-checkpoint-sha256 "$DRAFT_CHECKPOINT_SHA256")
fi
if [ -n "${RELEASE_ROOT:-}" ]; then
  arguments+=(--release-root "$RELEASE_ROOT")
fi
mkdir -p "$(dirname "$REPORT")"
"$PYTHON" "$REPOSITORY_ROOT/train/hf_dense_public_benchmark_pair.py" "${arguments[@]}"
chmod 444 "$REPORT"
