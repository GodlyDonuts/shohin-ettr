#!/bin/bash
#SBATCH --job-name=transfer-data-build
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/transfer_data_build_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/transfer_data_build_%j.err

set -euo pipefail

for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_REVISION DRAFT_ROOT \
  MERGED_DRAFTS DRAFT_RECEIPT PAIRS PAIRS_SHA256 MATH_BANK MATH_BANK_SHA256 \
  SCIENCE_BANK SCIENCE_BANK_SHA256 CODE_BANK CODE_BANK_SHA256 DATA_OUTPUT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test -x "$PYTHON"
for path in "$RUNTIME/pipeline/merge_idr1_drafts.py" \
  "$RUNTIME/pipeline/build_idr1_revision_data.py" \
  "$RUNTIME/pipeline/build_vcr1_revision_data.py" \
  "$RUNTIME/train/hf_pcj1_pairwise_judge.py" "$PAIRS" \
  "$MATH_BANK" "$SCIENCE_BANK" "$CODE_BANK"; do test -f "$path"; done
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$PAIRS" | cut -d' ' -f1)" = "$PAIRS_SHA256"
test "$(sha256sum "$MATH_BANK" | cut -d' ' -f1)" = "$MATH_BANK_SHA256"
test "$(sha256sum "$SCIENCE_BANK" | cut -d' ' -f1)" = "$SCIENCE_BANK_SHA256"
test "$(sha256sum "$CODE_BANK" | cut -d' ' -f1)" = "$CODE_BANK_SHA256"
test ! -e "$MERGED_DRAFTS"; test ! -e "$DRAFT_RECEIPT"; test ! -e "$DATA_OUTPUT"

REPORT_ARGS=()
for domain in math science; do
  for shard in $(seq 0 7); do
    path="$DRAFT_ROOT/draft_${domain}_s${shard}.report.json"
    test -f "$path"; REPORT_ARGS+=(--report "$path")
  done
done
test -f "$DRAFT_ROOT/draft_code_s0.report.json"
REPORT_ARGS+=(--report "$DRAFT_ROOT/draft_code_s0.report.json")

export PYTHONPATH="$RUNTIME/pipeline:$RUNTIME/train"
"$PYTHON" "$RUNTIME/pipeline/merge_idr1_drafts.py" \
  "${REPORT_ARGS[@]}" --bank "$MATH_BANK" --bank "$SCIENCE_BANK" \
  --bank "$CODE_BANK" --output "$MERGED_DRAFTS" --receipt "$DRAFT_RECEIPT" \
  --model-revision "$MODEL_REVISION" --base-model-draft
"$PYTHON" "$RUNTIME/pipeline/build_idr1_revision_data.py" \
  --pairs "$PAIRS" --drafts "$MERGED_DRAFTS" --draft-receipt "$DRAFT_RECEIPT" \
  --bank "$MATH_BANK" --bank "$SCIENCE_BANK" --bank "$CODE_BANK" \
  --output "$DATA_OUTPUT"
sha256sum "$MERGED_DRAFTS" "$DRAFT_RECEIPT" "$DATA_OUTPUT"/*.jsonl "$DATA_OUTPUT/report.json"
chmod a-w "$MERGED_DRAFTS" "$DRAFT_RECEIPT"; chmod -R a-w "$DATA_OUTPUT"

