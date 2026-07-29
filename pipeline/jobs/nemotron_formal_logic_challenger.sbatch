#!/bin/bash
#SBATCH -J shohin-nemo-logic
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/nemotron_formal_logic_challenger_%j.out
#SBATCH --open-mode=append
# Build a quarantined Formal Logic challenger from one exact pinned Parquet
# source. The result has zero training weight until residualization, holdouts,
# and a matched-token utility gate pass.
set -euo pipefail
export PATH=/usr/bin:/bin

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CODE_ROOT=${CODE_ROOT:?set CODE_ROOT to an isolated pinned source root}
TOKENIZE=$CODE_ROOT/pipeline/tokenize_shards.py
VERIFY_SHARDS=$CODE_ROOT/pipeline/verify_tokenized_shards.py
TOKENIZE_SHA256=${TOKENIZE_SHA256:?set exact tokenizer-pipeline SHA-256}
VERIFY_SHARDS_SHA256=${VERIFY_SHARDS_SHA256:?set exact shard-verifier SHA-256}
DATASET=nvidia/Nemotron-Pretraining-Specialized-v1.1
CONFIG=Nemotron-Pretraining-Formal-Logic
REVISION=13fa979be2e7f7e62913eee0ec5e97c8fd6e24af
RAW=${RAW:-$BASE/scratchpad/nemotron_specialized_v1_1_13fa979_profile_sources}
SOURCE=$RAW/formal_logic.parquet
CARD=$RAW/README.md
SOURCE_BYTES=73074239
SOURCE_SHA256=b8b641309dec27e836db6a2003d3045c6ed060c8d6a23e3a93100e1c8a3a450b
CARD_BYTES=7402
CARD_SHA256=93594fcc5d7e0a85bb8cbc6561b13d9d4c5c9cf4d21e7f81388829e86542e1bf
OUT=${OUT:-$BASE/artifacts/shards/nemotron_formal_logic_13fa979_challenger_128m}
TMP=${OUT}.partial
MAX_TOKENS=${MAX_TOKENS:-130000000}
TOKENIZER_BATCH_SIZE=${TOKENIZER_BATCH_SIZE:-256}

[[ -d "$CODE_ROOT" && ! -L "$CODE_ROOT" ]] || {
  echo "[formal-logic] isolated code root is unavailable" >&2
  exit 2
}
printf '%s  %s\n' \
  "$TOKENIZE_SHA256" "$TOKENIZE" \
  "$VERIFY_SHARDS_SHA256" "$VERIFY_SHARDS" | sha256sum -c -
[[ "$TOKENIZER_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || {
  echo "[formal-logic] tokenizer batch size differs" >&2
  exit 3
}
[[ "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] || {
  echo "[formal-logic] max tokens differs" >&2
  exit 3
}
[[ ! -e "$OUT" && ! -L "$OUT" ]] || {
  echo "[formal-logic] output exists: $OUT" >&2
  exit 4
}
[[ ! -e "$TMP" && ! -L "$TMP" ]] || {
  echo "[formal-logic] stale candidate partial exists: $TMP" >&2
  exit 4
}
for path in "$SOURCE" "$CARD"; do
  [[ -f "$path" && ! -L "$path" && "$(stat -c %a "$path")" == 444 ]] || {
    echo "[formal-logic] source custody differs: $path" >&2
    exit 5
  }
done
[[ "$(stat -c %s "$SOURCE")" == "$SOURCE_BYTES" ]] || {
  echo "[formal-logic] source byte size differs" >&2
  exit 5
}
[[ "$(stat -c %s "$CARD")" == "$CARD_BYTES" ]] || {
  echo "[formal-logic] dataset-card byte size differs" >&2
  exit 5
}
printf '%s  %s\n' \
  "$SOURCE_SHA256" "$SOURCE" \
  "$CARD_SHA256" "$CARD" | sha256sum -c -
[[ -s "$BASE/artifacts/shohin-tok-32k.json" ]] || {
  echo "[formal-logic] tokenizer is unavailable" >&2
  exit 6
}

cd "$CODE_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=true
"$PY" -u "$TOKENIZE" \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --dataset "$DATASET" \
  --config "$CONFIG" \
  --revision "$REVISION" \
  --input-files "$SOURCE" \
  --text-col text \
  --required-value license=cc-by-4.0 \
  --required-value metadata.category="$CONFIG" \
  --decontam-grams "$BASE/artifacts/evals/evalgrams.pkl" \
  --eval-glob "$BASE"/artifacts/evals/*.jsonl \
  --min-chars 300 \
  --max-chars 10000 \
  --exact-dedup \
  --max-line-repeat-fraction 0.18 \
  --max-boilerplate-markers 1 \
  --min-alpha-fraction 0.65 \
  --max-control-fraction 0.001 \
  --max-replacement-fraction 0.0005 \
  --out-dir "$TMP" \
  --shard-tokens 50000000 \
  --max-tokens "$MAX_TOKENS" \
  --tokenizer-batch-size "$TOKENIZER_BATCH_SIZE"

test -s "$TMP/manifest.json"
PYTHONPATH="$CODE_ROOT" "$PY" -m pipeline.verify_tokenized_shards \
  --shard-dir "$TMP" \
  --selection-code "$TOKENIZE" \
  --require-external-inputs
mv "$TMP" "$OUT"
find "$OUT" -type f -exec chmod 0444 {} +
find "$OUT" -type d -exec chmod 0555 {} +
echo "[formal-logic] immutable challenger committed at $OUT; not training-admitted"
