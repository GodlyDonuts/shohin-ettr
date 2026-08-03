#!/bin/bash
#SBATCH --job-name=ot-consensus
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/openthoughts_consensus_%j.out
#SBATCH --error=logs/openthoughts_consensus_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${DATASET_REVISION:?DATASET_REVISION is required}"
: "${EVAL_DIRS:?EVAL_DIRS is required}"
: "${OUTPUT:?OUTPUT is required}"
: "${REPORT:?REPORT is required}"

PYTHON=${PYTHON:-python3}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HOME
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

test -x "$(command -v "$PYTHON")" || test -x "$PYTHON"
test -f "$RUNTIME/pipeline/build_openthoughts_consensus_corpus.py"
test -f "$RUNTIME/pipeline/build_product_reasoning_seed_corpus.py"
test ! -e "$OUTPUT"
test ! -e "$REPORT"
mkdir -p logs "$(dirname "$OUTPUT")" "$(dirname "$REPORT")"

EVAL_ARGS=()
for directory in $EVAL_DIRS; do
  test -d "$directory"
  while IFS= read -r -d '' path; do
    EVAL_ARGS+=(--eval "$path")
  done < <(find "$directory" -maxdepth 1 -type f -name '*.jsonl' -print0 | sort -z)
done
if (( ${#EVAL_ARGS[@]} < 4 )); then
  echo "fewer than four evaluation manifests were found" >&2
  exit 3
fi

export PYTHONPATH="$RUNTIME/pipeline"
"$PYTHON" "$RUNTIME/pipeline/build_openthoughts_consensus_corpus.py" \
  --dataset-revision "$DATASET_REVISION" \
  --minimum-extracted 8 \
  --minimum-votes 8 \
  --minimum-fraction 0.60 \
  --minimum-margin 3 \
  "${EVAL_ARGS[@]}" \
  --output "$OUTPUT" \
  --report "$REPORT"
