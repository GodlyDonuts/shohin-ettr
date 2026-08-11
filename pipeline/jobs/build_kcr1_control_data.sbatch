#!/bin/bash
#SBATCH --job-name=kcr1-controls
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/kcr1_controls_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/kcr1_controls_%j.err

set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE SOURCE_SHA256 \
  SOURCE_REPORT SOURCE_REPORT_SHA256 MODEL_ROOT MODEL_MANIFEST_SHA256 OUTPUT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
test -x "$PYTHON"
test ! -e "$OUTPUT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = \
  "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$SOURCE" | cut -d' ' -f1)" = "$SOURCE_SHA256"
test "$(sha256sum "$SOURCE_REPORT" | cut -d' ' -f1)" = "$SOURCE_REPORT_SHA256"
test "$(sha256sum "$MODEL_ROOT/SHA256SUMS" | cut -d' ' -f1)" = \
  "$MODEL_MANIFEST_SHA256"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
"$PYTHON" "$RUNTIME/pipeline/build_kcr1_control_data.py" \
  --source "$SOURCE" --source-report "$SOURCE_REPORT" \
  --model-root "$MODEL_ROOT" --max-sequence-length 4096 --output "$OUTPUT"
sha256sum "$OUTPUT"/*.jsonl "$OUTPUT/report.json"
chmod -R a-w "$OUTPUT"
