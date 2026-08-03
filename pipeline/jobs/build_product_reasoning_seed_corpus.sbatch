#!/bin/bash
#SBATCH --job-name=product-corpus
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/product_corpus_%j.out
#SBATCH --error=logs/product_corpus_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${DATASET_REVISION:?DATASET_REVISION is required}"
: "${EVAL_DIR:?EVAL_DIR is required}"
: "${OUTPUT:?OUTPUT is required}"
: "${REPORT:?REPORT is required}"

PYTHON=${PYTHON:-python3}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false

test -f "$RUNTIME/pipeline/build_product_reasoning_seed_corpus.py"
test -f "$EVAL_DIR/gsm8k.jsonl"
test -f "$EVAL_DIR/math500.jsonl"
test ! -e "$OUTPUT"
test ! -e "$REPORT"
mkdir -p logs "$(dirname "$OUTPUT")" "$(dirname "$REPORT")"

EVAL_ARGS=()
while IFS= read -r -d '' path; do
  EVAL_ARGS+=(--eval "$path")
done < <(find "$EVAL_DIR" -maxdepth 1 -type f -name '*.jsonl' -print0 | sort -z)
if (( ${#EVAL_ARGS[@]} < 4 )); then
  echo "fewer than two evaluation manifests were found" >&2
  exit 3
fi

"$PYTHON" "$RUNTIME/pipeline/build_product_reasoning_seed_corpus.py" \
  --dataset-revision "$DATASET_REVISION" \
  --domain-cap math=60000 \
  --domain-cap code=30000 \
  --domain-cap science=30000 \
  "${EVAL_ARGS[@]}" \
  --output "$OUTPUT" \
  --report "$REPORT"
