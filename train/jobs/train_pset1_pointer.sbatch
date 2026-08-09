#!/bin/bash
#SBATCH --job-name=pset1-fit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/pset1_fit_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/pset1_fit_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION \
  MODEL_CONFIG_SHA256 HOST_CHECKPOINT HOST_CHECKPOINT_SHA256 DATA DATA_SHA256 \
  DATA_REPORT DATA_REPORT_SHA256 OUTPUT ARM; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
case "$ARM" in aligned|permuted) ;; *) exit 3 ;; esac
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
UPDATES=${UPDATES:-512}
test ! -e "$OUTPUT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$HOST_CHECKPOINT" | cut -d' ' -f1)" = "$HOST_CHECKPOINT_SHA256"
test "$(sha256sum "$DATA" | cut -d' ' -f1)" = "$DATA_SHA256"
test "$(sha256sum "$DATA_REPORT" | cut -d' ' -f1)" = "$DATA_REPORT_SHA256"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
nvidia-smi
"$PYTHON" "$RUNTIME/train/train_pset1_pointer.py" \
  --model-root "$MODEL_ROOT" --model-revision "$MODEL_REVISION" \
  --host-checkpoint "$HOST_CHECKPOINT" --data "$DATA" \
  --data-report "$DATA_REPORT" --output "$OUTPUT" --arm "$ARM" \
  --updates "$UPDATES" --learning-rate 3e-4 --seed 2026080917 --log-interval 8
test -f "$OUTPUT/checkpoint_$(printf '%07d' "$UPDATES").pt"
test -f "$OUTPUT/report.json"
chmod -R a-w "$OUTPUT"
