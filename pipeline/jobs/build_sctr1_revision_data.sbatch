#!/bin/bash
#SBATCH --job-name=sctr1-build
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/sctr1_build_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/sctr1_build_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${RUNTIME_MANIFEST_SHA256:?RUNTIME_MANIFEST_SHA256 is required}"
: "${PAIRS:?PAIRS is required}"
: "${PAIRS_SHA256:?PAIRS_SHA256 is required}"
: "${DRAFTS:?DRAFTS is required}"
: "${DRAFTS_SHA256:?DRAFTS_SHA256 is required}"
: "${DRAFT_RECEIPT:?DRAFT_RECEIPT is required}"
: "${DRAFT_RECEIPT_SHA256:?DRAFT_RECEIPT_SHA256 is required}"
: "${MATH_BANK:?MATH_BANK is required}"
: "${MATH_BANK_SHA256:?MATH_BANK_SHA256 is required}"
: "${SCIENCE_BANK:?SCIENCE_BANK is required}"
: "${SCIENCE_BANK_SHA256:?SCIENCE_BANK_SHA256 is required}"
: "${CODE_BANK:?CODE_BANK is required}"
: "${CODE_BANK_SHA256:?CODE_BANK_SHA256 is required}"
: "${OUTPUT:?OUTPUT is required}"

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

test -x "$PYTHON"
for path in \
  "$RUNTIME/pipeline/build_sctr1_revision_data.py" \
  "$RUNTIME/pipeline/build_idr1_revision_data.py" \
  "$RUNTIME/pipeline/build_vcr1_revision_data.py" \
  "$RUNTIME/train/hf_pcj1_pairwise_judge.py" \
  "$RUNTIME/train/sctr1_commit.py" \
  "$RUNTIME/train/ttr1_revision.py" \
  "$PAIRS" "$DRAFTS" "$DRAFT_RECEIPT" \
  "$MATH_BANK" "$SCIENCE_BANK" "$CODE_BANK"; do
  test -f "$path"
done
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = \
  "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
for expected in \
  "$PAIRS_SHA256 $PAIRS" \
  "$DRAFTS_SHA256 $DRAFTS" \
  "$DRAFT_RECEIPT_SHA256 $DRAFT_RECEIPT" \
  "$MATH_BANK_SHA256 $MATH_BANK" \
  "$SCIENCE_BANK_SHA256 $SCIENCE_BANK" \
  "$CODE_BANK_SHA256 $CODE_BANK"; do
  printf '%s\n' "$expected" | sha256sum -c - >/dev/null
done
test ! -e "$OUTPUT"

export PYTHONPATH="$RUNTIME/pipeline:$RUNTIME/train"
"$PYTHON" "$RUNTIME/pipeline/build_sctr1_revision_data.py" \
  --pairs "$PAIRS" \
  --drafts "$DRAFTS" \
  --draft-receipt "$DRAFT_RECEIPT" \
  --bank "$MATH_BANK" \
  --bank "$SCIENCE_BANK" \
  --bank "$CODE_BANK" \
  --output "$OUTPUT"

sha256sum \
  "$OUTPUT/train.jsonl" \
  "$OUTPUT/development.jsonl" \
  "$OUTPUT/holdout.jsonl" \
  "$OUTPUT/report.json"
chmod -R a-w "$OUTPUT"
