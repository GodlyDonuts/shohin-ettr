#!/bin/bash
#SBATCH -J shohin-fineweb-s4
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 3-00:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/fineweb_edu_score4_direct_%j.out
#SBATCH --open-mode=append
# Build a pinned score-4+ FineWeb-Edu candidate from physical Parquet files.
# This is not training admission: semantic review, cross-source residualization,
# a full audit, and a matched-token utility ablation remain mandatory.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CODE_ROOT=${CODE_ROOT:-$BASE/scratchpad/fineweb_edu_score4_code_v1}
TOKENIZE=$CODE_ROOT/pipeline/tokenize_shards.py
VERIFY_SHARDS=$CODE_ROOT/pipeline/verify_tokenized_shards.py
VERIFY_MANIFEST=$CODE_ROOT/pipeline/verify_pinned_hf_manifest.py
MANIFEST=$CODE_ROOT/pipeline/source_manifests/fineweb_edu_sample100bt_87f0914.json
TOKENIZE_SHA256=639a4934bae79e6301b4319ab96e26180fc7fe6bd85644789695d958e96cf48d
VERIFY_SHARDS_SHA256=efb9f1646d74e64c6243151fad716791c1817ead9a783b0f9ed99bdd56c54af6
VERIFY_MANIFEST_SHA256=5fee0bf17558a1e349c5fe492f9d8235431408b913f4cc5df91b817925c95287
MANIFEST_SHA256=759963e1db699dd27e0f1402d8f5622b22673a85ff6688999bf2e3afa5335fea
DATASET=HuggingFaceFW/fineweb-edu
REVISION=87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
RAW=${RAW:-$BASE/scratchpad/fineweb_edu_sample100bt_87f0914}
OUT=${OUT:-$BASE/artifacts/shards/fineweb_edu_score4_core_10b}
TMP=${OUT}.partial
HF_CURL_CONFIG=${HF_CURL_CONFIG:-$BASE/scratchpad/.hf_curl_auth.conf}
DOWNLOAD_CONCURRENCY=${DOWNLOAD_CONCURRENCY:-8}
MAX_TOKENS=${MAX_TOKENS:-10000000000}
DOMAIN_TOKEN_CAP=${DOMAIN_TOKEN_CAP:-5000000}

[[ -d "$CODE_ROOT" && ! -L "$CODE_ROOT" ]] || {
  echo "[fineweb-s4] isolated code root is unavailable" >&2
  exit 2
}
printf '%s  %s\n' \
  "$TOKENIZE_SHA256" "$TOKENIZE" \
  "$VERIFY_SHARDS_SHA256" "$VERIFY_SHARDS" \
  "$VERIFY_MANIFEST_SHA256" "$VERIFY_MANIFEST" \
  "$MANIFEST_SHA256" "$MANIFEST" | sha256sum -c -
[[ "$DOWNLOAD_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || {
  echo "[fineweb-s4] download concurrency differs" >&2
  exit 3
}
[[ ! -L "$HF_CURL_CONFIG" && -f "$HF_CURL_CONFIG" ]] || {
  echo "[fineweb-s4] Hugging Face curl config is unavailable" >&2
  exit 4
}
[[ "$(stat -c %a "$HF_CURL_CONFIG")" == 600 ]] || {
  echo "[fineweb-s4] Hugging Face curl config mode differs" >&2
  exit 4
}
[[ ! -e "$OUT" ]] || {
  echo "[fineweb-s4] output exists: $OUT" >&2
  exit 5
}
[[ ! -e "$TMP" ]] || {
  echo "[fineweb-s4] stale candidate partial exists: $TMP" >&2
  exit 5
}
[[ -s "$BASE/artifacts/shohin-tok-32k.json" ]] || {
  echo "[fineweb-s4] tokenizer is unavailable" >&2
  exit 6
}

mkdir -p "$RAW"
[[ -d "$RAW" && ! -L "$RAW" ]] || {
  echo "[fineweb-s4] source directory differs" >&2
  exit 7
}

download_one() {
  local relative_path=$1
  local expected_bytes=$2
  local expected_sha256=$3
  local name=${relative_path##*/}
  local final=$RAW/$name
  local partial=$final.part
  local url
  if [[ -e "$final" ]]; then
    [[ ! -L "$final" && -f "$final" ]] || {
      echo "[fineweb-s4] source is not a physical file: $final" >&2
      return 8
    }
  else
    if [[ -e "$partial" && "$(stat -c %s "$partial")" -gt "$expected_bytes" ]]; then
      echo "[fineweb-s4] oversized partial source: $partial" >&2
      return 9
    fi
    url="https://huggingface.co/datasets/$DATASET/resolve/$REVISION/$relative_path?download=true"
    curl \
      --config "$HF_CURL_CONFIG" \
      --fail \
      --location \
      --retry 8 \
      --retry-connrefused \
      --retry-delay 10 \
      --continue-at - \
      --silent \
      --show-error \
      --output "$partial" \
      "$url"
    [[ "$(stat -c %s "$partial")" == "$expected_bytes" ]] || {
      echo "[fineweb-s4] partial source byte size differs: $partial" >&2
      return 10
    }
    printf '%s  %s\n' "$expected_sha256" "$partial" | sha256sum -c -
    mv -- "$partial" "$final"
    chmod 0444 "$final"
  fi
  [[ "$(stat -c %s "$final")" == "$expected_bytes" ]] || {
    echo "[fineweb-s4] source byte size differs: $final" >&2
    return 11
  }
  printf '%s  %s\n' "$expected_sha256" "$final" | sha256sum -c -
}
export -f download_one
export DATASET REVISION RAW HF_CURL_CONFIG

cd "$CODE_ROOT"
"$PY" "$VERIFY_MANIFEST" \
  --manifest "$MANIFEST" \
  --dataset "$DATASET" \
  --revision "$REVISION" \
  --emit-tsv |
  xargs -r -P "$DOWNLOAD_CONCURRENCY" -n 3 \
    bash -c 'download_one "$@"' _

mapfile -t input_files < <(find "$RAW" -maxdepth 1 -type f -name '*.parquet' -print | sort)
[[ "${#input_files[@]}" == 140 ]] || {
  echo "[fineweb-s4] downloaded source count differs" >&2
  exit 12
}
if find "$RAW" -maxdepth 1 -type f -name '*.part' -print -quit | grep -q .; then
  echo "[fineweb-s4] source partials remain after download" >&2
  exit 12
fi

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
"$PY" -u "$TOKENIZE" \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --dataset "$DATASET" \
  --config sample-100BT \
  --revision "$REVISION" \
  --input-files "${input_files[@]}" \
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
  --max-line-repeat-fraction 0.30 \
  --max-boilerplate-markers 1 \
  --min-alpha-fraction 0.55 \
  --max-control-fraction 0.001 \
  --max-replacement-fraction 0.0005 \
  --domain-field url \
  --max-tokens-per-domain "$DOMAIN_TOKEN_CAP" \
  --out-dir "$TMP" \
  --shard-tokens 100000000 \
  --max-tokens "$MAX_TOKENS"

test -s "$TMP/manifest.json"
PYTHONPATH="$CODE_ROOT" "$PY" -m pipeline.verify_tokenized_shards \
  --shard-dir "$TMP" \
  --selection-code "$TOKENIZE" \
  --require-external-inputs
mv "$TMP" "$OUT"
find "$OUT" -type f -exec chmod 0444 {} +
find "$OUT" -type d -exec chmod 0555 {} +
echo "[fineweb-s4] immutable candidate committed at $OUT; not training-admitted"
