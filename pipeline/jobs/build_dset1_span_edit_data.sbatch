#!/bin/bash
#SBATCH --job-name=dset1-data
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/dset1_data_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/dset1_data_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 BASE_RUNTIME BASE_RUNTIME_MANIFEST_SHA256 \
  DSEO_TRAIN DSEO_TRAIN_SHA256 DSEO_DIAGNOSTIC DSEO_DIAGNOSTIC_SHA256 \
  DSEO_REPORT DSEO_REPORT_SHA256 MODEL_ROOT MODEL_CONFIG_SHA256 OUTPUT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test ! -e "$OUTPUT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
test "$(sha256sum "$BASE_RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$BASE_RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
(cd "$BASE_RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$DSEO_TRAIN" | cut -d' ' -f1)" = "$DSEO_TRAIN_SHA256"
test "$(sha256sum "$DSEO_DIAGNOSTIC" | cut -d' ' -f1)" = "$DSEO_DIAGNOSTIC_SHA256"
test "$(sha256sum "$DSEO_REPORT" | cut -d' ' -f1)" = "$DSEO_REPORT_SHA256"
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline:$BASE_RUNTIME/train:$BASE_RUNTIME/pipeline"
"$PYTHON" "$RUNTIME/pipeline/build_dset1_span_edit_data.py" \
  --dseo-train "$DSEO_TRAIN" --dseo-diagnostic "$DSEO_DIAGNOSTIC" \
  --dseo-report "$DSEO_REPORT" --model-root "$MODEL_ROOT" --output "$OUTPUT" \
  --max-sequence-length 4096 --max-script-tokens 32 \
  --min-train-sources 8000 --min-diagnostic-sources 1000
test -f "$OUTPUT/report.json"
chmod -R a-w "$OUTPUT"
