#!/bin/bash
#SBATCH --job-name=idr-scale-build
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/idr_scale_build_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/idr_scale_build_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${RUNTIME_MANIFEST_SHA256:?RUNTIME_MANIFEST_SHA256 is required}"
: "${MODEL_REVISION:?MODEL_REVISION is required}"
: "${DRAFT_CHECKPOINT_SHA256:?DRAFT_CHECKPOINT_SHA256 is required}"
: "${DRAFT_ROOT:?DRAFT_ROOT is required}"
: "${PAIRS:?PAIRS is required}"
: "${MATH_BANK:?MATH_BANK is required}"
: "${SCIENCE_BANK:?SCIENCE_BANK is required}"
: "${CODE_BANK:?CODE_BANK is required}"
: "${MERGED_ROOT:?MERGED_ROOT is required}"
: "${DATA_ROOT:?DATA_ROOT is required}"

test ! -e "$MERGED_ROOT"
test ! -e "$DATA_ROOT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)

for path in \
  "$RUNTIME/pipeline/merge_idr1_drafts.py" \
  "$RUNTIME/pipeline/build_idr1_revision_data.py" \
  "$PAIRS" "$MATH_BANK" "$SCIENCE_BANK" "$CODE_BANK"; do
  test -f "$path"
done

report_args=()
for domain in math science; do
  for shard in $(seq 0 7); do
    report="$DRAFT_ROOT/draft_${domain}_s${shard}.report.json"
    test -f "$report"
    report_args+=(--report "$report")
  done
done
report="$DRAFT_ROOT/draft_code_s0.report.json"
test -f "$report"
report_args+=(--report "$report")
test "${#report_args[@]}" -eq 34

mkdir -p "$MERGED_ROOT"
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
python3 "$RUNTIME/pipeline/merge_idr1_drafts.py" \
  "${report_args[@]}" \
  --bank "$MATH_BANK" \
  --bank "$SCIENCE_BANK" \
  --bank "$CODE_BANK" \
  --output "$MERGED_ROOT/merged_drafts.jsonl" \
  --receipt "$MERGED_ROOT/receipt.json" \
  --model-revision "$MODEL_REVISION" \
  --adapter-sha256 "$DRAFT_CHECKPOINT_SHA256"

python3 "$RUNTIME/pipeline/build_idr1_revision_data.py" \
  --pairs "$PAIRS" \
  --drafts "$MERGED_ROOT/merged_drafts.jsonl" \
  --draft-receipt "$MERGED_ROOT/receipt.json" \
  --bank "$MATH_BANK" \
  --bank "$SCIENCE_BANK" \
  --bank "$CODE_BANK" \
  --output "$DATA_ROOT"

sha256sum \
  "$MERGED_ROOT/merged_drafts.jsonl" \
  "$MERGED_ROOT/receipt.json" \
  "$DATA_ROOT/train.jsonl" \
  "$DATA_ROOT/development.jsonl" \
  "$DATA_ROOT/holdout.jsonl" \
  "$DATA_ROOT/report.json"
chmod -R a-w "$MERGED_ROOT" "$DATA_ROOT"
