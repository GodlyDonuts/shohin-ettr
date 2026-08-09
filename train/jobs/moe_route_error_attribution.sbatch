#!/bin/bash
#SBATCH --job-name=moe-route-attr
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/moe_route_attr_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/moe_route_attr_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION \
  MODEL_CONFIG_SHA256 MODEL_MANIFEST_SHA256 DATA DATA_SHA256 ATTRIBUTION \
  ATTRIBUTION_SHA256 MTR_CHECKPOINT MTR_CHECKPOINT_SHA256 RCR_CHECKPOINT \
  RCR_CHECKPOINT_SHA256 RANK1_CHECKPOINT RANK1_CHECKPOINT_SHA256 OUTPUT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$MODEL_ROOT/SHA256SUMS" | cut -d' ' -f1)" = "$MODEL_MANIFEST_SHA256"
(cd "$MODEL_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$DATA" | cut -d' ' -f1)" = "$DATA_SHA256"
test "$(sha256sum "$ATTRIBUTION" | cut -d' ' -f1)" = "$ATTRIBUTION_SHA256"
test "$(sha256sum "$MTR_CHECKPOINT" | cut -d' ' -f1)" = "$MTR_CHECKPOINT_SHA256"
test "$(sha256sum "$RCR_CHECKPOINT" | cut -d' ' -f1)" = "$RCR_CHECKPOINT_SHA256"
test "$(sha256sum "$RANK1_CHECKPOINT" | cut -d' ' -f1)" = "$RANK1_CHECKPOINT_SHA256"
test ! -e "$OUTPUT"
mkdir -p "$(dirname "$OUTPUT")"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
"$PYTHON" "$RUNTIME/train/moe_route_error_attribution.py" \
  --model-root "$MODEL_ROOT" --model-source-root "$MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" --development-data "$DATA" \
  --error-attribution "$ATTRIBUTION" --mtr-checkpoint "$MTR_CHECKPOINT" \
  --rcr-checkpoint "$RCR_CHECKPOINT" --rank1-checkpoint "$RANK1_CHECKPOINT" \
  --output "$OUTPUT"
sha256sum "$OUTPUT"
chmod a-w "$OUTPUT"
