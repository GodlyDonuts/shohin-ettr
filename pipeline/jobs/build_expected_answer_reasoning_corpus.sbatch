#!/bin/bash
#SBATCH --job-name=expected-corpus
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/expected_corpus_%j.out
#SBATCH --error=logs/expected_corpus_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${DATASET_ID:?DATASET_ID is required}"
: "${DATASET_REVISION:?DATASET_REVISION is required}"
: "${CONFIG:?CONFIG is required}"
: "${SPLIT:?SPLIT is required}"
: "${PROMPT_FIELD:?PROMPT_FIELD is required}"
: "${RESPONSE_FIELD:?RESPONSE_FIELD is required}"
: "${ANSWER_FIELD:?ANSWER_FIELD is required}"
: "${DOMAIN:?DOMAIN is required}"
: "${EVAL_DIRS:?EVAL_DIRS is required}"
: "${OUTPUT:?OUTPUT is required}"
: "${REPORT:?REPORT is required}"

PYTHON=${PYTHON:-python3}
MAXIMUM_ROWS=${MAXIMUM_ROWS:-500000}
SOURCE_FIELD=${SOURCE_FIELD:-}
LICENSE_FIELD=${LICENSE_FIELD:-}
METADATA_FIELDS=${METADATA_FIELDS:-}
REQUIRE_FIELD_EQUALS=${REQUIRE_FIELD_EQUALS:-}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HOME TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

test -f "$RUNTIME/pipeline/build_expected_answer_reasoning_corpus.py"
test ! -e "$OUTPUT"
test ! -e "$REPORT"
mkdir -p logs "$(dirname "$OUTPUT")" "$(dirname "$REPORT")"

EVAL_ARGS=()
for directory in $EVAL_DIRS; do
  test -d "$directory"
  while IFS= read -r -d '' path; do EVAL_ARGS+=(--eval "$path"); done \
    < <(find "$directory" -maxdepth 1 -type f -name '*.jsonl' -print0 | sort -z)
done
EXTRA_ARGS=()
if [[ -n "$SOURCE_FIELD" ]]; then EXTRA_ARGS+=(--source-field "$SOURCE_FIELD"); fi
if [[ -n "$LICENSE_FIELD" ]]; then EXTRA_ARGS+=(--license-field "$LICENSE_FIELD"); fi
for field in $METADATA_FIELDS; do EXTRA_ARGS+=(--metadata-field "$field"); done
for equality in $REQUIRE_FIELD_EQUALS; do EXTRA_ARGS+=(--require-field-equals "$equality"); done

export PYTHONPATH="$RUNTIME/pipeline"
"$PYTHON" "$RUNTIME/pipeline/build_expected_answer_reasoning_corpus.py" \
  --dataset-id "$DATASET_ID" --dataset-revision "$DATASET_REVISION" \
  --config "$CONFIG" --split "$SPLIT" \
  --prompt-field "$PROMPT_FIELD" --response-field "$RESPONSE_FIELD" \
  --answer-field "$ANSWER_FIELD" --domain "$DOMAIN" \
  --maximum-rows "$MAXIMUM_ROWS" "${EXTRA_ARGS[@]}" "${EVAL_ARGS[@]}" \
  --output "$OUTPUT" --report "$REPORT"
