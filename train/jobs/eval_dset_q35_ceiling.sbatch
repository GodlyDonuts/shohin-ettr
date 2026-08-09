#!/bin/bash
#SBATCH --job-name=dset-q35
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/dset_q35_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/dset_q35_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_CONFIG_SHA256 \
  DSET_DATA DSET_DATA_SHA256 PSET_DATA PSET_DATA_SHA256 PSET_REPORT \
  PSET_REPORT_SHA256 OUTPUT SHARD_INDEX SHARD_COUNT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test ! -e "$OUTPUT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$DSET_DATA" | cut -d' ' -f1)" = "$DSET_DATA_SHA256"
test "$(sha256sum "$PSET_DATA" | cut -d' ' -f1)" = "$PSET_DATA_SHA256"
test "$(sha256sum "$PSET_REPORT" | cut -d' ' -f1)" = "$PSET_REPORT_SHA256"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
nvidia-smi
"$PYTHON" "$RUNTIME/train/eval_dset_q35_ceiling.py" \
  --model-root "$MODEL_ROOT" --dset-data "$DSET_DATA" \
  --pset-data "$PSET_DATA" --pset-report "$PSET_REPORT" --output "$OUTPUT" \
  --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT" --seed 2026080918
test -f "$OUTPUT"
chmod a-w "$OUTPUT"
