#!/bin/bash
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue

set -euo pipefail
: "${REPOSITORY_ROOT:?}" "${PYTHON:?}" "${MODEL_REPOSITORY:?}"
: "${MODEL_REVISION:?}" "${MODEL_CONFIG_SHA256:?}" "${MODEL_ROOT:?}" "${MODEL_RECEIPT:?}"
: "${MODEL_EXPECTED_BYTES_MAX:?}"
read -r quota_used quota_hard file_used file_hard <<EOF
$(lfs quota -u sa305415 /lustre/fs1 | awk '$1 == "/lustre/fs1" {print $2, $4, $6, $8}')
EOF
projected_kib=$(( (MODEL_EXPECTED_BYTES_MAX + 1023) / 1024 ))
if [ $((quota_hard - quota_used - projected_kib)) -lt 134217728 ]; then
  echo "dense-public: restoration would breach the 128-GiB durable reserve" >&2
  exit 2
fi
if [ $((file_hard - file_used)) -lt 151000 ]; then
  echo "dense-public: restoration would breach the 150k-inode durable reserve" >&2
  exit 2
fi
export HF_HOME="${TMPDIR:-/tmp}/dense-restore-hf-${SLURM_JOB_ID}"
export PYTHONDONTWRITEBYTECODE=1
PYTHONPATH="$REPOSITORY_ROOT/pipeline:$REPOSITORY_ROOT/train" "$PYTHON" \
  "$REPOSITORY_ROOT/pipeline/restore_dense_public_model.py" \
  --repository "$MODEL_REPOSITORY" \
  --revision "$MODEL_REVISION" \
  --config-sha256 "$MODEL_CONFIG_SHA256" \
  --model-root "$MODEL_ROOT" \
  --receipt "$MODEL_RECEIPT"
