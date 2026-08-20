#!/bin/bash
set -euo pipefail

: "${RULER_ROOT:?}" "${MODEL_ROOT:?}" "${OUTPUT_ROOT:?}"
PYTHON=${PYTHON:?}
NUM_SAMPLES=${NUM_SAMPLES:-50}

if [ -e "$OUTPUT_ROOT" ]; then
  echo "refusing to replace RULER output root: $OUTPUT_ROOT" >&2
  exit 2
fi
if [ "$NUM_SAMPLES" -le 0 ]; then
  echo "RULER sample count must be positive" >&2
  exit 2
fi

tasks=(
  niah_single_1 niah_single_2 niah_single_3
  niah_multikey_1 niah_multikey_2 niah_multikey_3
  niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2
)
lengths=(4096 8192 16384 32768)
stage="${OUTPUT_ROOT}.stage.${SLURM_JOB_ID:-manual}"
test ! -e "$stage"
mkdir -m 700 -p "$stage"
trap 'rm -rf -- "$stage"' EXIT

json_root="$RULER_ROOT/scripts/data/synthetic/json"

"$PYTHON" - <<'PY'
import nltk
import numpy
import scipy
import tenacity
import transformers
import wonderwords
import yaml
PY
(
  cd "$json_root"
  if [ ! -s PaulGrahamEssays.json ]; then
    "$PYTHON" download_paulgraham_essay.py
  fi
  if [ ! -s squad.json ] || [ ! -s hotpotqa.json ]; then
    bash download_qa_dataset.sh
  fi
)

export PATH="$(dirname "$PYTHON"):/usr/bin:/bin"
export PYTHONDONTWRITEBYTECODE=1
for length in "${lengths[@]}"; do
  save_dir="$stage/${length}"
  for task in "${tasks[@]}"; do
    "$PYTHON" "$RULER_ROOT/scripts/data/prepare.py" \
      --save_dir "$save_dir" \
      --benchmark synthetic \
      --task "$task" \
      --subset validation \
      --tokenizer_path "$MODEL_ROOT" \
      --tokenizer_type hf \
      --max_seq_length "$length" \
      --model_template_type base \
      --num_samples "$NUM_SAMPLES" \
      --random_seed 42
    rows=$(wc -l < "$save_dir/$task/validation.jsonl")
    if [ "$rows" -ne "$NUM_SAMPLES" ]; then
      echo "RULER $length/$task cardinality differs: $rows" >&2
      exit 2
    fi
  done
done

find "$stage" -type d -exec chmod 555 {} +
find "$stage" -type f -exec chmod 444 {} +
mv "$stage" "$OUTPUT_ROOT"
trap - EXIT
printf 'RULER screen prepared: %s rows across %s lengths and %s tasks\n' \
  "$((NUM_SAMPLES * ${#lengths[@]} * ${#tasks[@]}))" \
  "${#lengths[@]}" "${#tasks[@]}"
