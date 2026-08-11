#!/bin/bash
#SBATCH --job-name=vte1-data
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/vte1_data_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/vte1_data_%j.err

set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE SOURCE_SHA256 \
  SOURCE_REPORT SOURCE_REPORT_SHA256 DRAFT_ROOT MODEL_ROOT \
  MODEL_MANIFEST_SHA256 ADAPTER_CHECKPOINT_SHA256 OUTPUT; do
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

draft_args=()
for shard in $(seq 0 15); do
  report=$DRAFT_ROOT/shard_${shard}.report.json
  test -f "$report"
  draft_args+=(--draft-report "$report")
done

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$RUNTIME/train:$RUNTIME/pipeline
"$PYTHON" "$RUNTIME/pipeline/build_vte1_equivalence_data.py" \
  --source "$SOURCE" \
  --source-report "$SOURCE_REPORT" \
  "${draft_args[@]}" \
  --adapter-checkpoint-sha256 "$ADAPTER_CHECKPOINT_SHA256" \
  --model-root "$MODEL_ROOT" \
  --max-sequence-length 4096 \
  --output "$OUTPUT"

sha256sum "$OUTPUT/train.jsonl" "$OUTPUT/report.json"
chmod -R a-w "$OUTPUT"
