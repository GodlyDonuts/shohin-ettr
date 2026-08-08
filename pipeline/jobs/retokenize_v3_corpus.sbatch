#!/bin/bash
# Retokenize one verified v3 corpus with a hash-bound immutable runtime.
#SBATCH -J shohin-retok-v3
#SBATCH -p normal
#SBATCH -c 64
#SBATCH --mem=192G
#SBATCH -t 2-00:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/retokenize_v3_%j.out
#SBATCH --open-mode=append

set -euo pipefail
set -o noclobber
export PATH=/usr/bin:/bin
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export TOKENIZERS_PARALLELISM=true

BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
PY=${PY:-$BASE/miniforge3/bin/python}
CODE_ROOT=${CODE_ROOT:?set an immutable retokenization runtime}
RUNTIME_SHA256SUMS=${RUNTIME_SHA256SUMS:?set the runtime SHA256SUMS path}
RUNTIME_SHA256SUMS_SHA256=${RUNTIME_SHA256SUMS_SHA256:?set its SHA-256}
SOURCE_DIR=${SOURCE_DIR:?set the verified source corpus}
SOURCE_SELECTION_CODE=${SOURCE_SELECTION_CODE:?set its exact selection code}
SOURCE_SELECTION_CODE_SHA256=${SOURCE_SELECTION_CODE_SHA256:?set its SHA-256}
SOURCE_MANIFEST_PAYLOAD_SHA256=${SOURCE_MANIFEST_PAYLOAD_SHA256:?bind source payload}
TARGET_TOKENIZER=${TARGET_TOKENIZER:?set the target tokenizer}
TARGET_TOKENIZER_SHA256=${TARGET_TOKENIZER_SHA256:?bind target tokenizer}
TARGET_EOS_TOKEN=${TARGET_EOS_TOKEN:-<|endoftext|>}
OUTPUT_DIR=${OUTPUT_DIR:?set a fresh output corpus}
REPORT=${REPORT:?set a fresh report path outside the corpus}
SHARD_TOKENS=${SHARD_TOKENS:-100000000}
BATCH_SIZE=${BATCH_SIZE:-1024}
MATERIALIZER=$CODE_ROOT/pipeline/retokenize_v3_corpus.py

for path in \
  "$CODE_ROOT" "$RUNTIME_SHA256SUMS" "$SOURCE_DIR" \
  "$SOURCE_SELECTION_CODE" "$TARGET_TOKENIZER" "$OUTPUT_DIR" "$REPORT"; do
  [[ "$path" == /* ]] || {
    echo "[retokenize-v3] paths must be absolute: $path" >&2
    exit 2
  }
done
[[ "$SHARD_TOKENS" =~ ^[1-9][0-9]*$ \
  && "$BATCH_SIZE" =~ ^[1-9][0-9]*$ \
  && -d "$CODE_ROOT" && ! -L "$CODE_ROOT" \
  && -f "$RUNTIME_SHA256SUMS" && ! -L "$RUNTIME_SHA256SUMS" \
  && -d "$SOURCE_DIR" && ! -L "$SOURCE_DIR" \
  && -f "$SOURCE_DIR/manifest.json" && ! -L "$SOURCE_DIR/manifest.json" \
  && -f "$SOURCE_SELECTION_CODE" && ! -L "$SOURCE_SELECTION_CODE" \
  && -f "$TARGET_TOKENIZER" && ! -L "$TARGET_TOKENIZER" \
  && -f "$MATERIALIZER" && ! -L "$MATERIALIZER" \
  && -x "$PY" \
  && ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" \
  && ! -e "$REPORT" && ! -L "$REPORT" ]] || {
  echo "[retokenize-v3] preflight differs" >&2
  exit 2
}
printf '%s  %s\n' \
  "$RUNTIME_SHA256SUMS_SHA256" "$RUNTIME_SHA256SUMS" \
  "$SOURCE_SELECTION_CODE_SHA256" "$SOURCE_SELECTION_CODE" \
  "$TARGET_TOKENIZER_SHA256" "$TARGET_TOKENIZER" | sha256sum -c -
"$PY" - "$SOURCE_DIR/manifest.json" "$SOURCE_MANIFEST_PAYLOAD_SHA256" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="ascii"))
if value.get("payload_sha256") != sys.argv[2]:
    raise SystemExit("[retokenize-v3] source manifest payload differs")
PY
(cd "$CODE_ROOT" && sha256sum -c "$RUNTIME_SHA256SUMS")

mkdir -p "$(dirname "$OUTPUT_DIR")" "$(dirname "$REPORT")"
export PYTHONPATH="$CODE_ROOT"
cd "$CODE_ROOT"
"$PY" -u "$MATERIALIZER" \
  --source-dir "$SOURCE_DIR" \
  --source-selection-code "$SOURCE_SELECTION_CODE" \
  --target-tokenizer "$TARGET_TOKENIZER" \
  --target-eos-token "$TARGET_EOS_TOKEN" \
  --selection-code "$MATERIALIZER" \
  --output-dir "$OUTPUT_DIR" \
  --report "$REPORT" \
  --shard-tokens "$SHARD_TOKENS" \
  --batch-size "$BATCH_SIZE"
"$PY" - "$REPORT" <<'PY'
import hashlib
import json
import sys

path = sys.argv[1]
value = json.load(open(path, encoding="ascii"))
claimed = value.pop("payload_sha256", None)
payload = json.dumps(
    value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
).encode("ascii")
if value.get("schema") != "shohin-v3-retokenization-v1" or (
    hashlib.sha256(payload).hexdigest() != claimed
):
    raise SystemExit("[retokenize-v3] report differs")
PY
chmod 0444 "$REPORT"
find "$OUTPUT_DIR" -type f -exec chmod 0444 {} +
find "$OUTPUT_DIR" -depth -type d -exec chmod 0555 {} +
echo "[retokenize-v3] complete; corpus remains quarantined pending final audits"
