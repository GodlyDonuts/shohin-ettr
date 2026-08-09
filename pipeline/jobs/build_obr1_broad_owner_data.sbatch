#!/bin/bash
#SBATCH --job-name=obr1-data
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/obr1_data_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/obr1_data_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE SOURCE_REPORT DEVELOPMENT DEVELOPMENT_REPORT MODEL_ROOT OUTPUT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test ! -e "$OUTPUT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
export TOKENIZERS_PARALLELISM=false PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
"$PYTHON" "$RUNTIME/pipeline/build_obr1_broad_owner_data.py" \
  --source "$SOURCE" --source-report "$SOURCE_REPORT" \
  --development "$DEVELOPMENT" --development-report "$DEVELOPMENT_REPORT" \
  --model-root "$MODEL_ROOT" --output "$OUTPUT" --ngram 13 \
  --max-sequence-length 4096
chmod -R a-w "$OUTPUT"
