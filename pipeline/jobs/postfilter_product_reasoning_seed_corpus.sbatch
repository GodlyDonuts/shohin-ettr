#!/bin/bash
#SBATCH --job-name=product-corpus-postfilter
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/product_corpus_postfilter_%j.out
#SBATCH --error=logs/product_corpus_postfilter_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${SOURCE:?SOURCE is required}"
: "${OUTPUT:?OUTPUT is required}"
: "${REPORT:?REPORT is required}"
EVAL_DIRS=${EVAL_DIRS:-${EVAL_DIR:-}}
if [[ -z "$EVAL_DIRS" ]]; then
  echo "EVAL_DIRS or EVAL_DIR is required" >&2
  exit 2
fi

PYTHON=${PYTHON:-python3}
test -f "$RUNTIME/pipeline/postfilter_product_reasoning_seed_corpus.py"
test -f "$SOURCE"
test ! -e "$OUTPUT"
test ! -e "$REPORT"
mkdir -p logs "$(dirname "$OUTPUT")" "$(dirname "$REPORT")"

EVAL_ARGS=()
IFS=: read -r -a EVAL_DIRECTORY_LIST <<< "$EVAL_DIRS"
for eval_dir in "${EVAL_DIRECTORY_LIST[@]}"; do
  test -d "$eval_dir"
  while IFS= read -r -d '' path; do
    EVAL_ARGS+=(--eval "$path")
  done < <(find "$eval_dir" -maxdepth 1 -type f -name '*.jsonl' -print0 | sort -z)
done
if (( ${#EVAL_ARGS[@]} < 4 )); then
  echo "fewer than four evaluation manifests were found" >&2
  exit 3
fi

"$PYTHON" "$RUNTIME/pipeline/postfilter_product_reasoning_seed_corpus.py" \
  --source "$SOURCE" \
  "${EVAL_ARGS[@]}" \
  --output "$OUTPUT" \
  --report "$REPORT"
