#!/bin/bash
#SBATCH --job-name=idr-product-build
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/idr_product_build_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/idr_product_build_%j.err

set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
: "${RUNTIME:?}" "${RUNTIME_MANIFEST_SHA256:?}"
: "${SOURCE:?}" "${SOURCE_SHA256:?}" "${SOURCE_REPORT:?}" "${SOURCE_REPORT_SHA256:?}"
: "${DRAFT_SHARD_ROOT:?}" "${SHARD_COUNT:?}"
: "${DRAFTS_OUTPUT:?}" "${DRAFTS_REPORT:?}" "${REVISION_OUTPUT:?}" "${REVISION_REPORT:?}"

test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$SOURCE" | cut -d' ' -f1)" = "$SOURCE_SHA256"
test "$(sha256sum "$SOURCE_REPORT" | cut -d' ' -f1)" = "$SOURCE_REPORT_SHA256"
for output in "$DRAFTS_OUTPUT" "$DRAFTS_REPORT" "$REVISION_OUTPUT" "$REVISION_REPORT"; do
  test ! -e "$output"
done

candidate_args=()
for shard in $(seq 0 $((SHARD_COUNT - 1))); do
  candidate_args+=(
    --candidate "$DRAFT_SHARD_ROOT/shard_$shard.candidates.jsonl"
    --candidate-report "$DRAFT_SHARD_ROOT/shard_$shard.report.json"
  )
done
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
"$PYTHON" "$RUNTIME/pipeline/build_aqc1_product.py" merge \
  --source "$SOURCE" --source-report "$SOURCE_REPORT" --input-stage source \
  "${candidate_args[@]}" --lineage draft --output "$DRAFTS_OUTPUT" --report "$DRAFTS_REPORT"
"$PYTHON" "$RUNTIME/pipeline/build_aqc1_product.py" revision \
  --source "$SOURCE" --source-report "$SOURCE_REPORT" \
  --drafts "$DRAFTS_OUTPUT" --drafts-report "$DRAFTS_REPORT" \
  --output "$REVISION_OUTPUT" --report "$REVISION_REPORT"
sha256sum "$DRAFTS_OUTPUT" "$DRAFTS_REPORT" "$REVISION_OUTPUT" "$REVISION_REPORT"
chmod a-w "$DRAFTS_OUTPUT" "$DRAFTS_REPORT" "$REVISION_OUTPUT" "$REVISION_REPORT"
