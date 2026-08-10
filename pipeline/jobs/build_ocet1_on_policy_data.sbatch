#!/bin/bash
#SBATCH --job-name=ocet1-materialize
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/ocet1_materialize_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/ocet1_materialize_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_CONFIG_SHA256 \
  SOURCE SOURCE_SHA256 SOURCE_REPORT SOURCE_REPORT_SHA256 PROPOSAL_ROOT OUTPUT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test ! -e "$OUTPUT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$SOURCE" | cut -d' ' -f1)" = "$SOURCE_SHA256"
test "$(sha256sum "$SOURCE_REPORT" | cut -d' ' -f1)" = "$SOURCE_REPORT_SHA256"
mapfile -t proposals < <(find "$PROPOSAL_ROOT" -maxdepth 1 -type f -name 'shard_*.json' | sort)
test "${#proposals[@]}" -eq 16
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONPATH="$BASE/train:$BASE/pipeline"
"$PYTHON" "$RUNTIME/pipeline/build_ocet1_on_policy_data.py" \
  --model-root "$MODEL_ROOT" --source "$SOURCE" --source-report "$SOURCE_REPORT" \
  --proposals "${proposals[@]}" --output "$OUTPUT"
test -f "$OUTPUT/report.json"
chmod -R a-w "$OUTPUT"
