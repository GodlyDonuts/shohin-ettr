#!/bin/bash
#SBATCH -J shohin-finepdf-arm
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=72G
#SBATCH -t 1-00:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/finepdfs_edu_policy_arm_%j.out
#SBATCH --open-mode=append
# Materialize one equal-token FinePDF policy arm from already verified physical
# sources. Output remains a quarantined ablation candidate, never an admitted
# training corpus.
set -euo pipefail
export PATH=/usr/bin:/bin

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CODE_ROOT=${CODE_ROOT:?set CODE_ROOT to an isolated pinned source root}
TOKENIZE=$CODE_ROOT/pipeline/tokenize_shards.py
VERIFY_SHARDS=$CODE_ROOT/pipeline/verify_tokenized_shards.py
POLICY=$CODE_ROOT/pipeline/finepdf_core_policy.py
TOKENIZE_SHA256=${TOKENIZE_SHA256:?set exact tokenizer-pipeline SHA-256}
VERIFY_SHARDS_SHA256=${VERIFY_SHARDS_SHA256:?set exact shard-verifier SHA-256}
POLICY_SHA256=${POLICY_SHA256:?set exact FinePDF policy SHA-256}
BASE_CANDIDATE=${BASE_CANDIDATE:-$BASE/artifacts/shards/finepdfs_edu_eng_core_10b}
BASE_CANDIDATE_PAYLOAD_SHA256=${BASE_CANDIDATE_PAYLOAD_SHA256:?set exact base candidate payload SHA-256}
ARM_TIER=${ARM_TIER:?set core or residual}
OUT=${OUT:?set a no-replace quarantined output path}
TMP=${OUT}.partial
MAX_TOKENS=${MAX_TOKENS:-100000000}
TOKENIZER_BATCH_SIZE=${TOKENIZER_BATCH_SIZE:-256}
DOMAIN_TOKEN_CAP=${DOMAIN_TOKEN_CAP:-100000000}
DATASET=HuggingFaceFW/finepdfs-edu
CONFIG=eng_Latn
REVISION=9cfabe2127faca99b3d5c4dc6d1fcb397399ebde

[[ "$ARM_TIER" == core || "$ARM_TIER" == residual ]] || {
  echo "[finepdf-arm] only core and residual ablation arms are allowed" >&2
  exit 2
}
[[ "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] || {
  echo "[finepdf-arm] max tokens differs" >&2
  exit 2
}
[[ "$TOKENIZER_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || {
  echo "[finepdf-arm] tokenizer batch size differs" >&2
  exit 2
}
[[ -d "$CODE_ROOT" && ! -L "$CODE_ROOT" ]] || {
  echo "[finepdf-arm] isolated code root is unavailable" >&2
  exit 3
}
printf '%s  %s\n' \
  "$TOKENIZE_SHA256" "$TOKENIZE" \
  "$VERIFY_SHARDS_SHA256" "$VERIFY_SHARDS" \
  "$POLICY_SHA256" "$POLICY" | sha256sum -c -
[[ -d "$BASE_CANDIDATE" && ! -L "$BASE_CANDIDATE" ]] || {
  echo "[finepdf-arm] base candidate is unavailable" >&2
  exit 4
}
[[ ! -e "$OUT" && ! -L "$OUT" && ! -e "$TMP" && ! -L "$TMP" ]] || {
  echo "[finepdf-arm] output or partial already exists" >&2
  exit 5
}

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=true
verification=$(
  PYTHONPATH="$CODE_ROOT" "$PY" -m pipeline.verify_tokenized_shards \
    --shard-dir "$BASE_CANDIDATE" \
    --require-external-inputs
)
verified_payload=$(
  "$PY" -c 'import json,sys; print(json.load(sys.stdin)["manifest_payload_sha256"])' \
    <<<"$verification"
)
[[ "$verified_payload" == "$BASE_CANDIDATE_PAYLOAD_SHA256" ]] || {
  echo "[finepdf-arm] base candidate payload differs" >&2
  exit 6
}

BASE_MANIFEST=$BASE_CANDIDATE/manifest.json
mapfile -t input_files < <(
  "$PY" -c \
    'import json,sys; print(*[x["path"] for x in json.load(open(sys.argv[1]))["source_files"]], sep="\n")' \
    "$BASE_MANIFEST"
)
mapfile -t eval_files < <(
  "$PY" -c \
    'import json,sys; print(*[x["path"] for x in json.load(open(sys.argv[1]))["decontamination"]["eval_files"]], sep="\n")' \
    "$BASE_MANIFEST"
)
eval_pickle=$(
  "$PY" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["decontamination"]["pickle_path"])' \
    "$BASE_MANIFEST"
)
[[ "${#input_files[@]}" == 16 && "${#eval_files[@]}" -gt 0 ]] || {
  echo "[finepdf-arm] base external-input ledger differs" >&2
  exit 7
}

"$PY" -u "$TOKENIZE" \
  --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
  --dataset "$DATASET" \
  --config "$CONFIG" \
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
  --decontam-grams "$eval_pickle" \
  --eval-glob "${eval_files[@]}" \
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
  --document-policy finepdf_core_v1 \
  --document-policy-allowed-tier "$ARM_TIER" \
  --out-dir "$TMP" \
  --shard-tokens "$MAX_TOKENS" \
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
echo "[finepdf-arm] immutable $ARM_TIER ablation candidate committed at $OUT; not training-admitted"
