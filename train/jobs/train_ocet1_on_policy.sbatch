#!/bin/bash
#SBATCH --job-name=ocet1-fit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/ocet1_fit_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/ocet1_fit_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_CONFIG_SHA256 \
  BNB_ROOT BNB_MANIFEST_SHA256 DATA DATA_SHA256 DATA_REPORT DATA_REPORT_SHA256 \
  WARM_START WARM_START_SHA256 OUTPUT ARM UPDATES GRADIENT_ACCUMULATION; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test ! -e "$OUTPUT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$BNB_ROOT/SHA256SUMS" | cut -d' ' -f1)" = "$BNB_MANIFEST_SHA256"
(cd "$BNB_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$DATA" | cut -d' ' -f1)" = "$DATA_SHA256"
test "$(sha256sum "$DATA_REPORT" | cut -d' ' -f1)" = "$DATA_REPORT_SHA256"
test "$(sha256sum "$WARM_START" | cut -d' ' -f1)" = "$WARM_START_SHA256"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$BNB_ROOT:$RUNTIME/train:$RUNTIME/pipeline"
nvidia-smi
"$PYTHON" "$RUNTIME/train/train_dset_transfer.py" \
  --model-root "$MODEL_ROOT" --model-revision 995ad96e \
  --data "$DATA" --data-report "$DATA_REPORT" --warm-start-checkpoint "$WARM_START" \
  --output "$OUTPUT" --arm "$ARM" --updates "$UPDATES" \
  --gradient-accumulation "$GRADIENT_ACCUMULATION" \
  --max-sequence-length 4096 --learning-rate 5e-5 \
  --controlled-layers 16 --rank 18 --quantization nf4 \
  --seed 2026080920 --data-seed 2026080920
test -f "$OUTPUT/report.json"
chmod -R a-w "$OUTPUT"
