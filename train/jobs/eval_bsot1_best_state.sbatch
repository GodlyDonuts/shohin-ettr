#!/bin/bash
#SBATCH --job-name=bsot1-best
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/bsot1_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/bsot1_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 BNB_ROOT BNB_MANIFEST_SHA256 \
  MODEL_ROOT MODEL_CONFIG_SHA256 CHECKPOINT CHECKPOINT_SHA256 DATA DATA_SHA256 \
  DATA_REPORT DATA_REPORT_SHA256 ISET_MERGED ISET_MERGED_SHA256 OUTPUT ARM \
  SHARD_INDEX SHARD_COUNT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test ! -e "$OUTPUT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$BNB_ROOT/SHA256SUMS" | cut -d' ' -f1)" = "$BNB_MANIFEST_SHA256"
(cd "$BNB_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$CHECKPOINT" | cut -d' ' -f1)" = "$CHECKPOINT_SHA256"
test "$(sha256sum "$DATA" | cut -d' ' -f1)" = "$DATA_SHA256"
test "$(sha256sum "$DATA_REPORT" | cut -d' ' -f1)" = "$DATA_REPORT_SHA256"
test "$(sha256sum "$ISET_MERGED" | cut -d' ' -f1)" = "$ISET_MERGED_SHA256"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi
"$PYTHON" "$RUNTIME/train/eval_rift1_fixed_point.py" \
  --model-root "$MODEL_ROOT" --checkpoint "$CHECKPOINT" \
  --data "$DATA" --data-report "$DATA_REPORT" --iset-merged "$ISET_MERGED" \
  --output "$OUTPUT" --arm "$ARM" --proposal-arm aligned \
  --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT" --seed 2026081011
test -f "$OUTPUT"
chmod a-w "$OUTPUT"
