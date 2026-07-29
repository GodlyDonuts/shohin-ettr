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
CODE_ROOT=${CODE_ROOT:?set CODE_ROOT to an isolated, pinned worktree}
TOKENIZE=$CODE_ROOT/pipeline/tokenize_shards.py
VERIFY_SHARDS=$CODE_ROOT/pipeline/verify_tokenized_shards.py
VERIFY_MANIFEST=$CODE_ROOT/pipeline/verify_pinned_hf_manifest.py
TOKENIZE_SHA256=c7bb632ce769a795bb615cebcb264372e7b36996146cafc7ceb42bddd2a8d1ea
VERIFY_SHARDS_SHA256=efb9f1646d74e64c6243151fad716791c1817ead9a783b0f9ed99bdd56c54af6
VERIFY_MANIFEST_SHA256=5fee0bf17558a1e349c5fe492f9d8235431408b913f4cc5df91b817925c95287
OUT=${OUT:-$BASE/artifacts/shards/pes2o_selected_core_10b}
TMP=${OUT}.partial
RAW=${RAW:-$BASE/scratchpad/pes2o_uniform30_v1}
MANIFEST=$BASE/artifacts/source_manifests/pes2o_uniform30_v1.json
MANIFEST_SHA256=8d3923e0b65d0102d10647796e610747b18167419cd9fe13be12d482f2c13192
HF_CURL_CONFIG=${HF_CURL_CONFIG:-$BASE/scratchpad/.hf_curl_auth.conf}
MAX_TOKENS=${MAX_TOKENS:-10000000000}
REVISION=${REVISION:-2caeba1585aaafa3c965410d80af85b09ed11cd9}
PUBLISHER_TOKEN_CAP=${PUBLISHER_TOKEN_CAP:-1000000000}
TOKENIZER_BATCH_SIZE=${TOKENIZER_BATCH_SIZE:-1}

[ ! -e "$OUT" ] || { echo "[pes2o-core] output exists: $OUT" >&2; exit 2; }
[ ! -e "$TMP" ] || { echo "[pes2o-core] stale partial: $TMP" >&2; exit 3; }
[[ -d "$CODE_ROOT" && ! -L "$CODE_ROOT" ]] || {
  echo "[pes2o-core] isolated code root is unavailable" >&2
  exit 4
}
printf '%s  %s\n' \
  "$TOKENIZE_SHA256" "$TOKENIZE" \
  "$VERIFY_SHARDS_SHA256" "$VERIFY_SHARDS" \
  "$VERIFY_MANIFEST_SHA256" "$VERIFY_MANIFEST" | sha256sum -c -
[[ "$TOKENIZER_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || {
  echo "[pes2o-core] tokenizer batch size differs" >&2
  exit 4
}
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

cd "$CODE_ROOT"
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
  "$PY" "$VERIFY_MANIFEST" \
    --manifest "$MANIFEST" \
    --dataset common-pile/peS2o \
    --revision "$REVISION" \
    --emit-tsv
)

mapfile -t input_files < <(find "$RAW" -maxdepth 1 -type f -name '*.json.gz' -print | sort)
[[ "${#input_files[@]}" == 30 ]] || {
  echo "[pes2o-core] downloaded source count differs" >&2
  exit 7
}

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=true
"$PY" -u "$TOKENIZE" \
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
  --max-tokens "$MAX_TOKENS" \
  --tokenizer-batch-size "$TOKENIZER_BATCH_SIZE"

test -s "$TMP/manifest.json"
PYTHONPATH="$CODE_ROOT" "$PY" -m pipeline.verify_tokenized_shards \
  --shard-dir "$TMP" \
  --selection-code "$TOKENIZE" \
  --require-external-inputs
mv "$TMP" "$OUT"
echo "[pes2o-core] candidate committed at $OUT; not training-admitted"
