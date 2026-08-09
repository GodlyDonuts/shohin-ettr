#!/bin/bash
#SBATCH --job-name=dseo1-force
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/dseo1_force_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/dseo1_force_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION \
  MODEL_CONFIG_SHA256 DATA DATA_SHA256 DATA_REPORT DATA_REPORT_SHA256 \
  CHECKPOINT CHECKPOINT_SHA256 BASELINE BASELINE_SHA256 OUTPUT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test ! -e "$OUTPUT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$DATA" | cut -d' ' -f1)" = "$DATA_SHA256"
test "$(sha256sum "$DATA_REPORT" | cut -d' ' -f1)" = "$DATA_REPORT_SHA256"
test "$(sha256sum "$CHECKPOINT" | cut -d' ' -f1)" = "$CHECKPOINT_SHA256"
test "$(sha256sum "$BASELINE" | cut -d' ' -f1)" = "$BASELINE_SHA256"
mkdir -p "$(dirname "$OUTPUT")" logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
nvidia-smi
"$PYTHON" "$RUNTIME/train/eval_dseo1_action_intervention.py" \
  --model-root "$MODEL_ROOT" --model-source-root "$MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" --adapter-checkpoint "$CHECKPOINT" \
  --data "$DATA" --data-report "$DATA_REPORT" --baseline "$BASELINE" \
  --output "$OUTPUT" --pairs 128 --batch-size 8 --max-new-tokens 768
