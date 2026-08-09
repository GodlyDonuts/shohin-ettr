#!/bin/bash
#SBATCH --job-name=dset1-fit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/dset1_fit_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/dset1_fit_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION \
  MODEL_CONFIG_SHA256 DATA DATA_SHA256 DATA_REPORT DATA_REPORT_SHA256 \
  OWNER_CHECKPOINT OWNER_CHECKPOINT_SHA256 OUTPUT ARM; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
case "$ARM" in aligned|swapped|hidden) ;; *) exit 3 ;; esac
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
UPDATES=${UPDATES:-512}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-$UPDATES}
test ! -e "$OUTPUT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$DATA" | cut -d' ' -f1)" = "$DATA_SHA256"
test "$(sha256sum "$DATA_REPORT" | cut -d' ' -f1)" = "$DATA_REPORT_SHA256"
test "$(sha256sum "$OWNER_CHECKPOINT" | cut -d' ' -f1)" = "$OWNER_CHECKPOINT_SHA256"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
nvidia-smi
"$PYTHON" "$RUNTIME/train/train_dset1_span_edit.py" \
  --model-root "$MODEL_ROOT" --model-source-root "$MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" --owner-checkpoint "$OWNER_CHECKPOINT" \
  --data "$DATA" --data-report "$DATA_REPORT" --output "$OUTPUT" --arm "$ARM" \
  --updates "$UPDATES" --gradient-accumulation 8 --max-sequence-length 4096 \
  --learning-rate 2e-5 --seed 2026080916 --data-seed 2026080916 \
  --log-interval 8 --checkpoint-interval "$CHECKPOINT_INTERVAL"
test -f "$OUTPUT/checkpoint_$(printf '%07d' "$UPDATES").pt"
test -f "$OUTPUT/report.json"
chmod -R a-w "$OUTPUT"
