#!/bin/bash
#SBATCH --job-name=sctr1-base-build
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/sctr1_base_build_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/sctr1_base_build_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${RUNTIME_MANIFEST_SHA256:?RUNTIME_MANIFEST_SHA256 is required}"
: "${MODEL_REVISION:?MODEL_REVISION is required}"
: "${DRAFT_ROOT:?DRAFT_ROOT is required}"
: "${MERGED_DRAFTS:?MERGED_DRAFTS is required}"
: "${DRAFT_RECEIPT:?DRAFT_RECEIPT is required}"
: "${PAIRS:?PAIRS is required}"
: "${PAIRS_SHA256:?PAIRS_SHA256 is required}"
: "${MATH_BANK:?MATH_BANK is required}"
: "${MATH_BANK_SHA256:?MATH_BANK_SHA256 is required}"
: "${SCIENCE_BANK:?SCIENCE_BANK is required}"
: "${SCIENCE_BANK_SHA256:?SCIENCE_BANK_SHA256 is required}"
: "${CODE_BANK:?CODE_BANK is required}"
: "${CODE_BANK_SHA256:?CODE_BANK_SHA256 is required}"
: "${SCTR_OUTPUT:?SCTR_OUTPUT is required}"
: "${STANDARD_OUTPUT:?STANDARD_OUTPUT is required}"

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

test -x "$PYTHON"
for path in \
  "$RUNTIME/pipeline/merge_idr1_drafts.py" \
  "$RUNTIME/pipeline/build_idr1_revision_data.py" \
  "$RUNTIME/pipeline/build_sctr1_revision_data.py" \
  "$RUNTIME/pipeline/build_vcr1_revision_data.py" \
  "$RUNTIME/train/hf_pcj1_pairwise_judge.py" \
  "$RUNTIME/train/sctr1_commit.py" \
  "$RUNTIME/train/ttr1_revision.py" \
  "$PAIRS" "$MATH_BANK" "$SCIENCE_BANK" "$CODE_BANK"; do
  test -f "$path"
done
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = \
  "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
for expected in \
  "$PAIRS_SHA256 $PAIRS" \
  "$MATH_BANK_SHA256 $MATH_BANK" \
  "$SCIENCE_BANK_SHA256 $SCIENCE_BANK" \
  "$CODE_BANK_SHA256 $CODE_BANK"; do
  printf '%s\n' "$expected" | sha256sum -c - >/dev/null
done
test ! -e "$MERGED_DRAFTS"
test ! -e "$DRAFT_RECEIPT"
test ! -e "$SCTR_OUTPUT"
test ! -e "$STANDARD_OUTPUT"

REPORT_ARGS=()
for domain in math science; do
  for shard in $(seq 0 7); do
    path="$DRAFT_ROOT/draft_${domain}_s${shard}.report.json"
    test -f "$path"
    REPORT_ARGS+=(--report "$path")
  done
done
test -f "$DRAFT_ROOT/draft_code_s0.report.json"
REPORT_ARGS+=(--report "$DRAFT_ROOT/draft_code_s0.report.json")

export PYTHONPATH="$RUNTIME/pipeline:$RUNTIME/train"
"$PYTHON" "$RUNTIME/pipeline/merge_idr1_drafts.py" \
  "${REPORT_ARGS[@]}" \
  --bank "$MATH_BANK" \
  --bank "$SCIENCE_BANK" \
  --bank "$CODE_BANK" \
  --output "$MERGED_DRAFTS" \
  --receipt "$DRAFT_RECEIPT" \
  --model-revision "$MODEL_REVISION" \
  --base-model-draft

COMMON_ARGS=(
  --pairs "$PAIRS"
  --drafts "$MERGED_DRAFTS"
  --draft-receipt "$DRAFT_RECEIPT"
  --bank "$MATH_BANK"
  --bank "$SCIENCE_BANK"
  --bank "$CODE_BANK"
)
"$PYTHON" "$RUNTIME/pipeline/build_sctr1_revision_data.py" \
  "${COMMON_ARGS[@]}" --output "$SCTR_OUTPUT"
"$PYTHON" "$RUNTIME/pipeline/build_idr1_revision_data.py" \
  "${COMMON_ARGS[@]}" --output "$STANDARD_OUTPUT"

sha256sum \
  "$MERGED_DRAFTS" "$DRAFT_RECEIPT" \
  "$SCTR_OUTPUT/train.jsonl" "$SCTR_OUTPUT/development.jsonl" \
  "$SCTR_OUTPUT/holdout.jsonl" "$SCTR_OUTPUT/report.json" \
  "$STANDARD_OUTPUT/train.jsonl" "$STANDARD_OUTPUT/development.jsonl" \
  "$STANDARD_OUTPUT/holdout.jsonl" "$STANDARD_OUTPUT/report.json"
chmod a-w "$MERGED_DRAFTS" "$DRAFT_RECEIPT"
chmod -R a-w "$SCTR_OUTPUT" "$STANDARD_OUTPUT"
