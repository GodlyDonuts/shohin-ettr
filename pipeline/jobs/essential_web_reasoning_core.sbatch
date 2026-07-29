#!/bin/bash
#SBATCH -J shohin-essential-core
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=72G
#SBATCH -t 3-00:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/essential_web_reasoning_core_%j.out
#SBATCH --open-mode=append
# Build a recent, reasoning-dense Essential-Web candidate from a deterministic
# pinned physical subset. The result is quarantined until residualization,
# private review, full audits, and an equal-token utility gate all pass.
set -euo pipefail
export PATH=/usr/bin:/bin

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CODE_ROOT=${CODE_ROOT:?set CODE_ROOT to an isolated pinned source root}
TOKENIZE=$CODE_ROOT/pipeline/tokenize_shards.py
VERIFY_SHARDS=$CODE_ROOT/pipeline/verify_tokenized_shards.py
VERIFY_MANIFEST=$CODE_ROOT/pipeline/verify_pinned_hf_manifest.py
MANIFEST=$CODE_ROOT/pipeline/source_manifests/essential_web_2024_38_uniform256_ce4eccc.json
TOKENIZE_SHA256=${TOKENIZE_SHA256:?set exact tokenizer-pipeline SHA-256}
VERIFY_SHARDS_SHA256=${VERIFY_SHARDS_SHA256:?set exact shard-verifier SHA-256}
VERIFY_MANIFEST_SHA256=${VERIFY_MANIFEST_SHA256:?set exact manifest-verifier SHA-256}
MANIFEST_SHA256=${MANIFEST_SHA256:?set exact source-manifest SHA-256}
DATASET=EssentialAI/essential-web-v1.0
REVISION=ce4eccc7e9604667b6d7f32cb6274b8b41f3113d
RAW=${RAW:-$BASE/scratchpad/essential_web_2024_38_uniform256_ce4eccc}
OUT=${OUT:-$BASE/artifacts/shards/essential_web_reasoning_core_1b}
TMP=${OUT}.partial
HF_CURL_CONFIG=${HF_CURL_CONFIG:-$BASE/scratchpad/.hf_curl_auth.conf}
DOWNLOAD_CONCURRENCY=${DOWNLOAD_CONCURRENCY:-8}
MAX_TOKENS=${MAX_TOKENS:-1000000000}
DOMAIN_TOKEN_CAP=${DOMAIN_TOKEN_CAP:-5000000}
TOKENIZER_BATCH_SIZE=${TOKENIZER_BATCH_SIZE:-256}

[[ -d "$CODE_ROOT" && ! -L "$CODE_ROOT" ]] || {
  echo "[essential] isolated code root is unavailable" >&2
  exit 2
}
printf '%s  %s\n' \
  "$TOKENIZE_SHA256" "$TOKENIZE" \
  "$VERIFY_SHARDS_SHA256" "$VERIFY_SHARDS" \
  "$VERIFY_MANIFEST_SHA256" "$VERIFY_MANIFEST" \
  "$MANIFEST_SHA256" "$MANIFEST" | sha256sum -c -
[[ "$DOWNLOAD_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || {
  echo "[essential] download concurrency differs" >&2
  exit 3
}
[[ "$TOKENIZER_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || {
  echo "[essential] tokenizer batch size differs" >&2
  exit 3
}
[[ ! -L "$HF_CURL_CONFIG" && -f "$HF_CURL_CONFIG" ]] || {
  echo "[essential] Hugging Face curl config is unavailable" >&2
  exit 4
}
[[ "$(stat -c %a "$HF_CURL_CONFIG")" == 600 ]] || {
  echo "[essential] Hugging Face curl config mode differs" >&2
  exit 4
}
[[ ! -e "$OUT" && ! -L "$OUT" ]] || {
  echo "[essential] output exists: $OUT" >&2
  exit 5
}
[[ ! -e "$TMP" && ! -L "$TMP" ]] || {
  echo "[essential] stale candidate partial exists: $TMP" >&2
  exit 5
}
[[ -s "$BASE/artifacts/shohin-tok-32k.json" ]] || {
  echo "[essential] tokenizer is unavailable" >&2
  exit 6
}

mkdir -p "$RAW"
[[ -d "$RAW" && ! -L "$RAW" ]] || {
  echo "[essential] source directory differs" >&2
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
      echo "[essential] source is not a physical file: $final" >&2
      return 8
    }
  else
    if [[ -e "$partial" && "$(stat -c %s "$partial")" -gt "$expected_bytes" ]]; then
      echo "[essential] oversized partial source: $partial" >&2
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
      echo "[essential] partial source byte size differs: $partial" >&2
      return 10
    }
    printf '%s  %s\n' "$expected_sha256" "$partial" | sha256sum -c -
    mv -- "$partial" "$final"
    chmod 0444 "$final"
  fi
  [[ "$(stat -c %s "$final")" == "$expected_bytes" ]] || {
    echo "[essential] source byte size differs: $final" >&2
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
[[ "${#input_files[@]}" == 256 ]] || {
  echo "[essential] downloaded source count differs" >&2
  exit 12
}
if find "$RAW" -maxdepth 1 -type f -name '*.part' -print -quit | grep -q .; then
  echo "[essential] source partials remain after download" >&2
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
  --config default \
  --revision "$REVISION" \
  --input-files "${input_files[@]}" \
  --text-col text \
  --required-min-number quality_signals.fasttext.english=0.90 \
  --required-allowed-value eai_taxonomy.reasoning_depth.primary.code=3 \
  --required-allowed-value eai_taxonomy.reasoning_depth.primary.code=4 \
  --required-allowed-value eai_taxonomy.reasoning_depth.primary.code=5 \
  --required-allowed-value eai_taxonomy.technical_correctness.primary.code=4 \
  --required-allowed-value eai_taxonomy.technical_correctness.primary.code=5 \
  --required-allowed-value eai_taxonomy.extraction_artifacts.primary.code=0 \
  --required-allowed-value eai_taxonomy.missing_content.primary.code=0 \
  --required-allowed-value eai_taxonomy.document_type_v2.primary.code=3 \
  --required-allowed-value eai_taxonomy.document_type_v2.primary.code=8 \
  --required-allowed-value eai_taxonomy.document_type_v2.primary.code=9 \
  --required-allowed-value eai_taxonomy.document_type_v2.primary.code=10 \
  --required-allowed-value eai_taxonomy.document_type_v2.primary.code=18 \
  --required-allowed-value eai_taxonomy.document_type_v2.primary.code=20 \
  --required-allowed-value eai_taxonomy.document_type_v2.primary.code=21 \
  --required-allowed-value eai_taxonomy.document_type_v2.primary.code=23 \
  --decontam-grams "$BASE/artifacts/evals/evalgrams.pkl" \
  --eval-glob "$BASE"/artifacts/evals/*.jsonl \
  --min-chars 500 \
  --max-chars 100000 \
  --exact-dedup \
  --max-line-repeat-fraction 0.20 \
  --max-boilerplate-markers 1 \
  --min-alpha-fraction 0.55 \
  --max-control-fraction 0.001 \
  --max-replacement-fraction 0.0005 \
  --domain-field metadata.source_domain \
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
echo "[essential] immutable candidate committed at $OUT; not training-admitted"
