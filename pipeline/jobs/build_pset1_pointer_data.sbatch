#!/bin/bash
#SBATCH --job-name=pset1-data
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/pset1_data_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/pset1_data_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 DSET_TRAIN DSET_TRAIN_SHA256 \
  DSET_DIAGNOSTIC DSET_DIAGNOSTIC_SHA256 DSET_REPORT DSET_REPORT_SHA256 \
  MODEL_ROOT MODEL_CONFIG_SHA256 OUTPUT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test ! -e "$OUTPUT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$DSET_TRAIN" | cut -d' ' -f1)" = "$DSET_TRAIN_SHA256"
test "$(sha256sum "$DSET_DIAGNOSTIC" | cut -d' ' -f1)" = "$DSET_DIAGNOSTIC_SHA256"
test "$(sha256sum "$DSET_REPORT" | cut -d' ' -f1)" = "$DSET_REPORT_SHA256"
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONPATH="$RUNTIME/pipeline:$RUNTIME/train"
"$PYTHON" "$RUNTIME/pipeline/build_pset1_pointer_data.py" \
  --dset-train "$DSET_TRAIN" --dset-diagnostic "$DSET_DIAGNOSTIC" \
  --dset-report "$DSET_REPORT" --model-root "$MODEL_ROOT" --output "$OUTPUT" \
  --train-sources 4096 --diagnostic-sources 256 \
  --max-sequence-length 4096 --max-replacement-tokens 16
test -f "$OUTPUT/report.json"
chmod -R a-w "$OUTPUT"
