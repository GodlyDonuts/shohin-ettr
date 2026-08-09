#!/bin/bash
#SBATCH --job-name=ecr1-mech
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/ecr1_fit_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/ecr1_fit_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION \
  MODEL_CONFIG_SHA256 MODEL_MANIFEST_SHA256 DATA DATA_SHA256 OUTPUT MODE \
  DRAFT_CONTROL UPDATES RANK ALPHA; do
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
test ! -e "$OUTPUT"
LOCAL_PARENT=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
EFFECTIVE_MODEL_ROOT=$LOCAL_PARENT/model
mkdir -p "$EFFECTIVE_MODEL_ROOT" "$(dirname "$OUTPUT")"
cp -a "$MODEL_ROOT"/. "$EFFECTIVE_MODEL_ROOT"/
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
nvidia-smi
"$PYTHON" "$RUNTIME/train/train_ecr1_product.py" \
  --model-root "$EFFECTIVE_MODEL_ROOT" --model-source-root "$MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" --model-loader causal --data "$DATA" \
  --output "$OUTPUT" --architecture "${ARCHITECTURE:-ecr1}" \
  --mode "$MODE" --draft-control "$DRAFT_CONTROL" \
  --updates "$UPDATES" --rank "$RANK" --alpha "$ALPHA" \
  --batch-size "${BATCH_SIZE:-1}" --gradient-accumulation "${ACCUMULATION:-1}" \
  --max-rows "${MAX_ROWS:-8}" --max-sequence-length "${MAX_LENGTH:-4096}" \
  --learning-rate "${LEARNING_RATE:-2e-5}" \
  --controlled-layers "${CONTROLLED_LAYERS:-4}" \
  --revision-experts "${REVISION_EXPERTS:-4}" \
  --revision-top-k "${REVISION_TOP_K:-2}" \
  --balance-weight "${BALANCE_WEIGHT:-0.01}" \
  --state-width "${STATE_WIDTH:-64}" --head-width "${HEAD_WIDTH:-32}" \
  --router-scale "${ROUTER_SCALE:-1.0}" \
  --entropy-floor "${ENTROPY_FLOOR:-0.80}" \
  --collapse-weight "${COLLAPSE_WEIGHT:-0.01}" \
  --seed "${SEED:-2026080901}" --data-seed "${DATA_SEED:-2026080814}" \
  --log-interval "${LOG_INTERVAL:-1}" --checkpoint-interval "$UPDATES"
sha256sum "$OUTPUT/report.json" "$OUTPUT/checkpoint_$(printf '%07d' "$UPDATES").pt"
chmod -R a-w "$OUTPUT"
