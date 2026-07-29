#!/bin/bash
#SBATCH -J shohin-fineweb-core
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 3-00:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/fineweb_edu_core_%j.out
#SBATCH --open-mode=append
# Build a pinned score-4+ FineWeb-Edu core. This is a candidate artifact only:
# it still requires a shard scan, cross-corpus dedup, semantic review receipt,
# and equal-token ablation before training admission.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
OUT=${OUT:-$BASE/artifacts/shards/fineweb_edu_score4_core_10b}
TMP=${OUT}.partial
MAX_TOKENS=${MAX_TOKENS:-10000000000}
REVISION=${REVISION:-87f09149ef4734204d70ed1d046ddc9ca3f2b8f9}
DATASET_CONFIG=${DATASET_CONFIG:-sample-100BT}
DOMAIN_TOKEN_CAP=${DOMAIN_TOKEN_CAP:-5000000}

[ ! -e "$OUT" ] || { echo "[fineweb-core] output exists: $OUT" >&2; exit 2; }
[ ! -e "$TMP" ] || { echo "[fineweb-core] stale partial: $TMP" >&2; exit 3; }
[ -s "$BASE/artifacts/shohin-tok-32k.json" ] || {
  echo "[fineweb-core] tokenizer missing" >&2
  exit 4
}

cd "$BASE"
"$PY" pipeline/tokenize_shards.py \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --dataset HuggingFaceFW/fineweb-edu \
  --config "$DATASET_CONFIG" \
  --revision "$REVISION" \
  --text-col text \
  --lang en \
  --lang-field language \
  --require-lang-field \
  --min-number-field int_score \
  --min-number 4 \
  --decontam-grams "$BASE/artifacts/evals/evalgrams.pkl" \
  --eval-glob "$BASE"/artifacts/evals/*.jsonl \
  --min-chars 400 \
  --max-chars 100000 \
  --exact-dedup \
  --max-line-repeat-fraction 0.35 \
  --max-boilerplate-markers 1 \
  --domain-field url \
  --max-tokens-per-domain "$DOMAIN_TOKEN_CAP" \
  --out-dir "$TMP" \
  --shard-tokens 100000000 \
  --max-tokens "$MAX_TOKENS"

test -s "$TMP/manifest.json"
"$PY" -m pipeline.verify_tokenized_shards \
  --shard-dir "$TMP" \
  --selection-code "$BASE/pipeline/tokenize_shards.py" \
  --require-external-inputs
mv "$TMP" "$OUT"
echo "[fineweb-core] candidate committed at $OUT; not training-admitted"
