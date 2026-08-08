#!/bin/bash
#SBATCH --job-name=idr-product-merge
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/idr_product_merge_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/idr_product_merge_%j.err

set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
: "${RUNTIME:?}" "${RUNTIME_MANIFEST_SHA256:?}"
: "${SOURCE:?}" "${SOURCE_REPORT:?}" "${REVISION_DATA:?}" "${REVISION_REPORT:?}"
: "${TREATMENT_SHARD_ROOT:?}" "${CONTROL_SHARD_ROOT:?}" "${SHARD_COUNT:?}"
: "${TREATMENT_OUTPUT:?}" "${TREATMENT_REPORT:?}" "${CONTROL_OUTPUT:?}" "${CONTROL_REPORT:?}"
: "${COMPARISON_REPORT:?}"

test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
for output in "$TREATMENT_OUTPUT" "$TREATMENT_REPORT" "$CONTROL_OUTPUT" "$CONTROL_REPORT" "$COMPARISON_REPORT"; do
  test ! -e "$output"
done
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"

merge_lineage() {
  local lineage=$1 shard_root=$2 output=$3 report=$4
  local candidate_args=()
  for shard in $(seq 0 $((SHARD_COUNT - 1))); do
    candidate_args+=(
      --candidate "$shard_root/shard_$shard.candidates.jsonl"
      --candidate-report "$shard_root/shard_$shard.report.json"
    )
  done
  "$PYTHON" "$RUNTIME/pipeline/build_aqc1_product.py" merge \
    --source "$REVISION_DATA" --source-report "$REVISION_REPORT" --input-stage revision \
    "${candidate_args[@]}" --lineage "$lineage" --output "$output" --report "$report"
}

merge_lineage idr4 "$TREATMENT_SHARD_ROOT" "$TREATMENT_OUTPUT" "$TREATMENT_REPORT"
merge_lineage control "$CONTROL_SHARD_ROOT" "$CONTROL_OUTPUT" "$CONTROL_REPORT"
"$PYTHON" "$RUNTIME/pipeline/score_idr_product.py" \
  --source "$SOURCE" --source-report "$SOURCE_REPORT" \
  --treatment "$TREATMENT_OUTPUT" --treatment-report "$TREATMENT_REPORT" \
  --treatment-stage merged-idr4 \
  --control "$CONTROL_OUTPUT" --control-report "$CONTROL_REPORT" \
  --control-stage merged-control --report "$COMPARISON_REPORT"
sha256sum "$TREATMENT_OUTPUT" "$TREATMENT_REPORT" "$CONTROL_OUTPUT" "$CONTROL_REPORT" "$COMPARISON_REPORT"
chmod a-w "$TREATMENT_OUTPUT" "$TREATMENT_REPORT" "$CONTROL_OUTPUT" "$CONTROL_REPORT" "$COMPARISON_REPORT"
