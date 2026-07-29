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
RAW=${RAW:-$BASE/scratchpad/pes2o_uniform30_v1}
MANIFEST=$BASE/artifacts/source_manifests/pes2o_uniform30_v1.json
MANIFEST_SHA256=8d3923e0b65d0102d10647796e610747b18167419cd9fe13be12d482f2c13192
HF_CURL_CONFIG=${HF_CURL_CONFIG:-$BASE/scratchpad/.hf_curl_auth.conf}
MAX_TOKENS=${MAX_TOKENS:-10000000000}
REVISION=${REVISION:-2caeba1585aaafa3c965410d80af85b09ed11cd9}
PUBLISHER_TOKEN_CAP=${PUBLISHER_TOKEN_CAP:-1000000000}

[ ! -e "$OUT" ] || { echo "[pes2o-core] output exists: $OUT" >&2; exit 2; }
[ ! -e "$TMP" ] || { echo "[pes2o-core] stale partial: $TMP" >&2; exit 3; }
[ -s "$BASE/artifacts/shohin-tok-32k.json" ] || {
  echo "[pes2o-core] tokenizer missing" >&2
  exit 4
}
printf '%s  %s\n' "$MANIFEST_SHA256" "$MANIFEST" | sha256sum -c -
[[ ! -L "$HF_CURL_CONFIG" && -f "$HF_CURL_CONFIG" ]] || {
  echo "[pes2o-core] Hugging Face curl config is unavailable" >&2
  exit 5
}
[[ "$(stat -c %a "$HF_CURL_CONFIG")" == 600 ]] || {
  echo "[pes2o-core] Hugging Face curl config mode differs" >&2
  exit 5
}

cd "$BASE"
mkdir -p "$RAW"
while IFS=$'\t' read -r relative_path expected_bytes expected_sha256; do
  name=${relative_path##*/}
  final=$RAW/$name
  partial=$final.part
  if [[ -e "$final" ]]; then
    [[ ! -L "$final" && -f "$final" ]] || {
      echo "[pes2o-core] final source is not a physical file: $final" >&2
      exit 5
    }
  else
    url="https://huggingface.co/datasets/common-pile/peS2o/resolve/$REVISION/$relative_path?download=true"
    curl \
      --config "$HF_CURL_CONFIG" \
      --fail \
      --location \
      --retry 8 \
      --retry-connrefused \
      --retry-delay 10 \
      --continue-at - \
      --output "$partial" \
      "$url"
    [[ "$(stat -c %s "$partial")" == "$expected_bytes" ]] || {
      echo "[pes2o-core] partial source byte size differs: $partial" >&2
      exit 6
    }
    printf '%s  %s\n' "$expected_sha256" "$partial" | sha256sum -c -
    mv "$partial" "$final"
    chmod 0444 "$final"
  fi
  [[ "$(stat -c %s "$final")" == "$expected_bytes" ]] || {
    echo "[pes2o-core] source byte size differs: $final" >&2
    exit 8
  }
  printf '%s  %s\n' "$expected_sha256" "$final" | sha256sum -c -
done < <(
  "$PY" pipeline/verify_pinned_hf_manifest.py \
    --manifest "$MANIFEST" \
    --dataset common-pile/peS2o \
    --revision "$REVISION" \
    --emit-tsv
)

input_files=("$RAW"/*.json.gz)
[[ "${#input_files[@]}" == 30 ]] || {
  echo "[pes2o-core] downloaded source count differs" >&2
  exit 7
}

"$PY" pipeline/tokenize_shards.py \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --dataset common-pile/peS2o \
  --revision "$REVISION" \
  --input-files "${input_files[@]}" \
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
