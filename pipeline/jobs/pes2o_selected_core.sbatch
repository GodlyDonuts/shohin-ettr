#!/bin/bash
#SBATCH -J shohin-pes2o-core
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 3-00:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/pes2o_selected_core_%j.out
#SBATCH --open-mode=append
# Build a clean, open-license peS2o candidate. This is not training admission:
# human review, cross-source residualization, full shard audit, and a
# matched-token utility ablation remain mandatory.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
OUT=${OUT:-$BASE/artifacts/shards/pes2o_selected_core_10b}
TMP=${OUT}.partial
MAX_TOKENS=${MAX_TOKENS:-10000000000}
REVISION=${REVISION:-2caeba1585aaafa3c965410d80af85b09ed11cd9}
PUBLISHER_TOKEN_CAP=${PUBLISHER_TOKEN_CAP:-1000000000}

[ ! -e "$OUT" ] || { echo "[pes2o-core] output exists: $OUT" >&2; exit 2; }
[ ! -e "$TMP" ] || { echo "[pes2o-core] stale partial: $TMP" >&2; exit 3; }
[ -s "$BASE/artifacts/shohin-tok-32k.json" ] || {
  echo "[pes2o-core] tokenizer missing" >&2
  exit 4
}

cd "$BASE"
"$PY" pipeline/tokenize_shards.py \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --dataset common-pile/peS2o \
  --revision "$REVISION" \
  --text-col text \
  --allowed-values-field metadata.oa_license \
  --allowed-values CCBY CCBYSA pd \
  --decontam-grams "$BASE/artifacts/evals/evalgrams.pkl" \
  --eval-glob "$BASE"/artifacts/evals/*.jsonl \
  --min-chars 4000 \
  --max-chars 120000 \
  --exact-dedup \
  --max-line-repeat-fraction 0.12 \
  --max-boilerplate-markers 1 \
  --min-alpha-fraction 0.70 \
  --max-control-fraction 0.001 \
  --max-replacement-fraction 0.0005 \
  --domain-field metadata.oa_url \
  --max-tokens-per-domain "$PUBLISHER_TOKEN_CAP" \
  --out-dir "$TMP" \
  --shard-tokens 100000000 \
  --max-tokens "$MAX_TOKENS"

test -s "$TMP/manifest.json"
"$PY" -m pipeline.verify_tokenized_shards \
  --shard-dir "$TMP" \
  --selection-code "$BASE/pipeline/tokenize_shards.py" \
  --require-external-inputs
mv "$TMP" "$OUT"
echo "[pes2o-core] candidate committed at $OUT; not training-admitted"
