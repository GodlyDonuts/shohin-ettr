#!/bin/bash
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --no-requeue

set -euo pipefail
: "${REPOSITORY_ROOT:?}" "${PYTHON:?}" "${SOURCE_ROOT:?}" "${DATA_ROOT:?}"
: "${TRAIN_MATH:?}" "${TRAIN_SCIENCE:?}" "${TRAIN_CODE:?}"

test ! -e "$SOURCE_ROOT"
test ! -e "$DATA_ROOT"
source_stage="${SOURCE_ROOT}.stage.${SLURM_JOB_ID}"
data_stage="${DATA_ROOT}.stage.${SLURM_JOB_ID}"
trap 'rm -rf -- "$source_stage" "$data_stage"' EXIT
mkdir -m 700 -p "$source_stage"

git -C "$source_stage" clone --filter=blob:none --no-checkout \
  https://github.com/google-research/google-research.git google-research
git -C "$source_stage/google-research" sparse-checkout init --cone
git -C "$source_stage/google-research" sparse-checkout set instruction_following_eval
git -C "$source_stage/google-research" checkout --detach \
  589e977488f21a336a3d3da9b96da91ddbcf935e
test "$(git -C "$source_stage/google-research" rev-parse HEAD)" = \
  589e977488f21a336a3d3da9b96da91ddbcf935e

git -C "$source_stage" clone https://github.com/Zayne-sprague/MuSR.git musr
git -C "$source_stage/musr" checkout --detach \
  b1f4d4168a9cfc6760e8b74d728e4516023dfaa5
test "$(git -C "$source_stage/musr" rev-parse HEAD)" = \
  b1f4d4168a9cfc6760e8b74d728e4516023dfaa5
cat > "$source_stage/SOURCE_REVISIONS.json" <<'EOF'
{
  "google_research": "589e977488f21a336a3d3da9b96da91ddbcf935e",
  "musr": "b1f4d4168a9cfc6760e8b74d728e4516023dfaa5"
}
EOF
rm -rf -- "$source_stage/google-research/.git" "$source_stage/musr/.git"

export HF_HOME="${TMPDIR:-/tmp}/dense-public-hf-${SLURM_JOB_ID}"
export PYTHONDONTWRITEBYTECODE=1
PYTHONPATH="$REPOSITORY_ROOT/pipeline:$REPOSITORY_ROOT/train" "$PYTHON" \
  "$REPOSITORY_ROOT/pipeline/build_dense_public_benchmark_data.py" \
  --ifeval-input "$source_stage/google-research/instruction_following_eval/data/input_data.jsonl" \
  --musr-root "$source_stage/musr" \
  --training-source "$TRAIN_MATH" \
  --training-source "$TRAIN_SCIENCE" \
  --training-source "$TRAIN_CODE" \
  --output-root "$data_stage"

"$PYTHON" - "$data_stage/smoke.questions.jsonl" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

prompt = "Return only the word READY in uppercase."
row = {
    "schema": "shohin-dense-public-benchmark-question-v1",
    "id": hashlib.sha256(prompt.encode()).hexdigest(),
    "benchmark": "mmlu_pro",
    "upstream_id": "nonbenchmark-mechanics-smoke-v1",
    "question": prompt,
    "response_mode": "general",
}
Path(sys.argv[1]).write_text(json.dumps(row, sort_keys=True) + "\n")
PY

find "$source_stage" -type d -exec chmod 555 {} +
find "$source_stage" -type f -exec chmod 444 {} +
find "$data_stage" -type d -exec chmod 555 {} +
find "$data_stage" -type f -exec chmod 444 {} +
mv "$source_stage" "$SOURCE_ROOT"
mv "$data_stage" "$DATA_ROOT"
trap - EXIT
