#!/bin/bash
#SBATCH --job-name=iset1-data
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/iset1_data_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/iset1_data_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_CONFIG_SHA256 \
  TRAIN TRAIN_SHA256 DIAGNOSTIC DIAGNOSTIC_SHA256 SOURCE_REPORT \
  SOURCE_REPORT_SHA256 OUTPUT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test ! -e "$OUTPUT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$TRAIN" | cut -d' ' -f1)" = "$TRAIN_SHA256"
test "$(sha256sum "$DIAGNOSTIC" | cut -d' ' -f1)" = "$DIAGNOSTIC_SHA256"
test "$(sha256sum "$SOURCE_REPORT" | cut -d' ' -f1)" = "$SOURCE_REPORT_SHA256"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
"$PYTHON" "$RUNTIME/pipeline/build_iset1_data.py" \
  --model-root "$MODEL_ROOT" --train "$TRAIN" --diagnostic "$DIAGNOSTIC" \
  --source-report "$SOURCE_REPORT" --output "$OUTPUT" \
  --maximum 4096 --script-maximum 32
test -f "$OUTPUT/report.json"
chmod -R a-w "$OUTPUT"
