#!/bin/bash
#SBATCH --job-name=idr1-control-merge
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/idr1_control_merge_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/idr1_control_merge_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${RUNTIME_MANIFEST_SHA256:?RUNTIME_MANIFEST_SHA256 is required}"
: "${DATA_ROOT:?DATA_ROOT is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

test -x "$PYTHON"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = \
  "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test -f "$RUNTIME/pipeline/merge_idr1_evaluation_shards.py"

export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
for split in development holdout; do
  split_root="$OUTPUT_ROOT/$split"
  test ! -e "$split_root/candidates.jsonl"
  test ! -e "$split_root/report.json"
  "$PYTHON" "$RUNTIME/pipeline/merge_idr1_evaluation_shards.py" \
    --data "$DATA_ROOT/$split.jsonl" \
    --split "$split" \
    --shard-report "$split_root/shard_0/report.json" \
    --shard-report "$split_root/shard_1/report.json" \
    --shard-report "$split_root/shard_2/report.json" \
    --shard-report "$split_root/shard_3/report.json" \
    --candidates-output "$split_root/candidates.jsonl" \
    --report "$split_root/report.json"
  sha256sum "$split_root/candidates.jsonl" "$split_root/report.json"
  chmod a-w "$split_root/candidates.jsonl" "$split_root/report.json"
done
