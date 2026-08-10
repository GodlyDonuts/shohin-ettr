#!/bin/bash
#SBATCH --job-name=gset1-gate
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/gset1_gate_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/gset1_gate_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 FEATURE_DIR OUTPUT ARM; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test ! -e "$OUTPUT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
mapfile -t FEATURES < <(find "$FEATURE_DIR" -maxdepth 1 -type f -name 'feature_*.pt' | sort)
test "${#FEATURES[@]}" -eq 8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
nvidia-smi
"$PYTHON" "$RUNTIME/train/train_gset1_fault_gate.py" \
  --features "${FEATURES[@]}" --output "$OUTPUT" --arm "$ARM" \
  --updates 1024 --batch-pairs 256 --gate-width 256 --learning-rate 3e-4 \
  --margin 2.0 --seed 2026080922 --data-seed 2026080922
test -f "$OUTPUT/report.json"
chmod -R a-w "$OUTPUT"
