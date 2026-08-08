#!/bin/bash
#SBATCH --job-name=idr1-data-recovery
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/idr1_data_recovery_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/idr1_data_recovery_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${RUNTIME_MANIFEST_SHA256:?RUNTIME_MANIFEST_SHA256 is required}"
: "${MERGED_DRAFTS:?MERGED_DRAFTS is required}"
: "${MERGED_DRAFTS_SHA256:?MERGED_DRAFTS_SHA256 is required}"
: "${DRAFT_RECEIPT:?DRAFT_RECEIPT is required}"
: "${DRAFT_RECEIPT_SHA256:?DRAFT_RECEIPT_SHA256 is required}"
: "${PAIRS:?PAIRS is required}"
: "${PAIRS_SHA256:?PAIRS_SHA256 is required}"
: "${MATH_BANK:?MATH_BANK is required}"
: "${MATH_BANK_SHA256:?MATH_BANK_SHA256 is required}"
: "${SCIENCE_BANK:?SCIENCE_BANK is required}"
: "${SCIENCE_BANK_SHA256:?SCIENCE_BANK_SHA256 is required}"
: "${CODE_BANK:?CODE_BANK is required}"
: "${CODE_BANK_SHA256:?CODE_BANK_SHA256 is required}"
: "${DATA_OUTPUT:?DATA_OUTPUT is required}"

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

test -x "$PYTHON"
for path in \
  "$RUNTIME/pipeline/build_idr1_revision_data.py" \
  "$RUNTIME/pipeline/build_vcr1_revision_data.py" \
  "$RUNTIME/train/hf_pcj1_pairwise_judge.py" \
  "$RUNTIME/train/hf_cvg1_completion_verifier.py" \
  "$MERGED_DRAFTS" "$DRAFT_RECEIPT" "$PAIRS" \
  "$MATH_BANK" "$SCIENCE_BANK" "$CODE_BANK"; do
  test -f "$path"
done
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MERGED_DRAFTS" | cut -d' ' -f1)" = "$MERGED_DRAFTS_SHA256"
test "$(sha256sum "$DRAFT_RECEIPT" | cut -d' ' -f1)" = "$DRAFT_RECEIPT_SHA256"
test "$(sha256sum "$PAIRS" | cut -d' ' -f1)" = "$PAIRS_SHA256"
test "$(sha256sum "$MATH_BANK" | cut -d' ' -f1)" = "$MATH_BANK_SHA256"
test "$(sha256sum "$SCIENCE_BANK" | cut -d' ' -f1)" = "$SCIENCE_BANK_SHA256"
test "$(sha256sum "$CODE_BANK" | cut -d' ' -f1)" = "$CODE_BANK_SHA256"
test ! -e "$DATA_OUTPUT"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$RUNTIME/pipeline:$RUNTIME/train"
"$PYTHON" "$RUNTIME/pipeline/build_idr1_revision_data.py" \
  --pairs "$PAIRS" \
  --drafts "$MERGED_DRAFTS" \
  --draft-receipt "$DRAFT_RECEIPT" \
  --bank "$MATH_BANK" \
  --bank "$SCIENCE_BANK" \
  --bank "$CODE_BANK" \
  --output "$DATA_OUTPUT"

sha256sum \
  "$DATA_OUTPUT/train.jsonl" \
  "$DATA_OUTPUT/development.jsonl" \
  "$DATA_OUTPUT/holdout.jsonl" \
  "$DATA_OUTPUT/report.json"
chmod -R a-w "$DATA_OUTPUT"
