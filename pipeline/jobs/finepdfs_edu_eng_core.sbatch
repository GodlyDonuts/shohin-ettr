#!/bin/bash
#SBATCH -J shohin-finepdfs-eng
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=72G
#SBATCH -t 3-00:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/finepdfs_edu_eng_core_%j.out
#SBATCH --open-mode=append
# Build a language-stable, nontruncated English FinePDFs-Edu candidate from
# uniformly selected physical Parquet files. The result remains quarantined
# until semantic review, residualization, holdouts, audits, and utility gates.
set -euo pipefail
export PATH=/usr/bin:/bin

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CODE_ROOT=${CODE_ROOT:?set CODE_ROOT to an isolated pinned source root}
TOKENIZE=$CODE_ROOT/pipeline/tokenize_shards.py
VERIFY_SHARDS=$CODE_ROOT/pipeline/verify_tokenized_shards.py
VERIFY_MANIFEST=$CODE_ROOT/pipeline/verify_pinned_hf_manifest.py
MANIFEST=$CODE_ROOT/pipeline/source_manifests/finepdfs_edu_eng_uniform16_9cfabe2.json
TOKENIZE_SHA256=${TOKENIZE_SHA256:?set exact tokenizer-pipeline SHA-256}
VERIFY_SHARDS_SHA256=${VERIFY_SHARDS_SHA256:?set exact shard-verifier SHA-256}
VERIFY_MANIFEST_SHA256=${VERIFY_MANIFEST_SHA256:?set exact manifest-verifier SHA-256}
MANIFEST_SHA256=${MANIFEST_SHA256:?set exact source-manifest SHA-256}
DATASET=HuggingFaceFW/finepdfs-edu
REVISION=9cfabe2127faca99b3d5c4dc6d1fcb397399ebde
RAW=${RAW:-$BASE/scratchpad/finepdfs_edu_eng_uniform16_9cfabe2}
OUT=${OUT:-$BASE/artifacts/shards/finepdfs_edu_eng_core_10b}
TMP=${OUT}.partial
HF_CURL_CONFIG=${HF_CURL_CONFIG:-$BASE/scratchpad/.hf_curl_auth.conf}
DOWNLOAD_CONCURRENCY=${DOWNLOAD_CONCURRENCY:-8}
MAX_TOKENS=${MAX_TOKENS:-10000000000}
DOMAIN_TOKEN_CAP=${DOMAIN_TOKEN_CAP:-100000000}
TOKENIZER_BATCH_SIZE=${TOKENIZER_BATCH_SIZE:-256}

[[ -d "$CODE_ROOT" && ! -L "$CODE_ROOT" ]] || {
  echo "[finepdfs] isolated code root is unavailable" >&2
  exit 2
}
printf '%s  %s\n' \
  "$TOKENIZE_SHA256" "$TOKENIZE" \
  "$VERIFY_SHARDS_SHA256" "$VERIFY_SHARDS" \
  "$VERIFY_MANIFEST_SHA256" "$VERIFY_MANIFEST" \
  "$MANIFEST_SHA256" "$MANIFEST" | sha256sum -c -
[[ "$DOWNLOAD_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || {
  echo "[finepdfs] download concurrency differs" >&2
  exit 3
}
[[ "$TOKENIZER_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || {
  echo "[finepdfs] tokenizer batch size differs" >&2
  exit 3
}
[[ ! -L "$HF_CURL_CONFIG" && -f "$HF_CURL_CONFIG" ]] || {
  echo "[finepdfs] Hugging Face curl config is unavailable" >&2
  exit 4
}
[[ "$(stat -c %a "$HF_CURL_CONFIG")" == 600 ]] || {
  echo "[finepdfs] Hugging Face curl config mode differs" >&2
  exit 4
}
[[ ! -e "$OUT" && ! -L "$OUT" ]] || {
  echo "[finepdfs] output exists: $OUT" >&2
  exit 5
}
[[ ! -e "$TMP" && ! -L "$TMP" ]] || {
  echo "[finepdfs] stale candidate partial exists: $TMP" >&2
  exit 5
}
[[ -s "$BASE/artifacts/shohin-tok-32k.json" ]] || {
  echo "[finepdfs] tokenizer is unavailable" >&2
  exit 6
}

mkdir -p "$RAW"
[[ -d "$RAW" && ! -L "$RAW" ]] || {
  echo "[finepdfs] source directory differs" >&2
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
      echo "[finepdfs] source is not a physical file: $final" >&2
      return 8
    }
  else
    if [[ -e "$partial" && "$(stat -c %s "$partial")" -gt "$expected_bytes" ]]; then
      echo "[finepdfs] oversized partial source: $partial" >&2
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
      echo "[finepdfs] partial source byte size differs: $partial" >&2
      return 10
    }
    printf '%s  %s\n' "$expected_sha256" "$partial" | sha256sum -c -
    mv -- "$partial" "$final"
    chmod 0444 "$final"
  fi
  [[ "$(stat -c %s "$final")" == "$expected_bytes" ]] || {
    echo "[finepdfs] source byte size differs: $final" >&2
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

mapfile -t input_files < <(
  find "$RAW" -maxdepth 1 -type f -name '*.parquet' -print | sort
)
[[ "${#input_files[@]}" == 16 ]] || {
  echo "[finepdfs] downloaded source count differs" >&2
  exit 12
}
if find "$RAW" -maxdepth 1 -type f -name '*.part' -print -quit | grep -q .; then
  echo "[finepdfs] source partials remain after download" >&2
  exit 12
fi

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=true
"$PY" -u "$TOKENIZE" \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --dataset "$DATASET" \
  --config eng_Latn \
  --revision "$REVISION" \
  --input-files "${input_files[@]}" \
  --text-col text \
  --lang eng_Latn \
  --lang-field full_doc_lid \
  --require-lang-field \
  --required-value page_average_lid=eng_Latn \
  --required-value is_truncated=false \
  --min-number-field page_average_lid_score \
  --min-number 0.90 \
  --decontam-grams "$BASE/artifacts/evals/evalgrams.pkl" \
  --eval-glob "$BASE"/artifacts/evals/*.jsonl \
  --min-chars 1000 \
  --max-chars 200000 \
  --exact-dedup \
  --max-line-repeat-fraction 0.20 \
  --max-boilerplate-markers 1 \
  --min-alpha-fraction 0.65 \
  --max-control-fraction 0.001 \
  --max-replacement-fraction 0.0005 \
  --domain-field url \
  --max-tokens-per-domain "$DOMAIN_TOKEN_CAP" \
  --out-dir "$TMP" \
  --shard-tokens 100000000 \
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
echo "[finepdfs] immutable candidate committed at $OUT; not training-admitted"
