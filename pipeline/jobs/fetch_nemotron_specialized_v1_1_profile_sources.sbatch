#!/bin/bash
#SBATCH -J shohin-nemotron-v11-fetch
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/nemotron_v11_profile_fetch_%j.out
#SBATCH --open-mode=append
# Fetch only the two compact zero-weight profile challengers and their pinned
# dataset card. No tokenization or training occurs in this job.
set -euo pipefail
export PATH=/usr/bin:/bin

BASE=/lustre/fs1/home/sa305415/shohin
DATASET=nvidia/Nemotron-Pretraining-Specialized-v1.1
REVISION=13fa979be2e7f7e62913eee0ec5e97c8fd6e24af
RAW=${RAW:-$BASE/scratchpad/nemotron_specialized_v1_1_13fa979_profile_sources}
HF_CURL_CONFIG=${HF_CURL_CONFIG:-$BASE/scratchpad/.hf_curl_auth.conf}

[[ ! -L "$HF_CURL_CONFIG" && -f "$HF_CURL_CONFIG" ]] || {
  echo "[nemotron-v11] Hugging Face curl config is unavailable" >&2
  exit 2
}
[[ "$(stat -c %a "$HF_CURL_CONFIG")" == 600 ]] || {
  echo "[nemotron-v11] Hugging Face curl config mode differs" >&2
  exit 2
}
mkdir -p "$RAW"
[[ -d "$RAW" && ! -L "$RAW" ]] || {
  echo "[nemotron-v11] source directory differs" >&2
  exit 3
}

fetch_one() {
  local relative_path=$1
  local final_name=$2
  local expected_bytes=$3
  local expected_sha256=$4
  local final=$RAW/$final_name
  local partial=$final.part
  if [[ -e "$final" ]]; then
    [[ ! -L "$final" && -f "$final" ]] || {
      echo "[nemotron-v11] source is not a physical file: $final" >&2
      return 4
    }
  else
    if [[ -e "$partial" && "$(stat -c %s "$partial")" -gt "$expected_bytes" ]]; then
      echo "[nemotron-v11] oversized partial source: $partial" >&2
      return 5
    fi
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
      "https://huggingface.co/datasets/$DATASET/resolve/$REVISION/$relative_path?download=true"
    [[ "$(stat -c %s "$partial")" == "$expected_bytes" ]] || {
      echo "[nemotron-v11] partial source byte size differs: $partial" >&2
      return 6
    }
    printf '%s  %s\n' "$expected_sha256" "$partial" | sha256sum -c -
    mv -- "$partial" "$final"
    chmod 0444 "$final"
  fi
  [[ "$(stat -c %s "$final")" == "$expected_bytes" ]] || {
    echo "[nemotron-v11] source byte size differs: $final" >&2
    return 7
  }
  printf '%s  %s\n' "$expected_sha256" "$final" | sha256sum -c -
}

fetch_one \
  README.md \
  README.md \
  7402 \
  93594fcc5d7e0a85bb8cbc6561b13d9d4c5c9cf4d21e7f81388829e86542e1bf
fetch_one \
  Nemotron-Pretraining-Formal-Logic/part_000000.parquet \
  formal_logic.parquet \
  73074239 \
  b8b641309dec27e836db6a2003d3045c6ed060c8d6a23e3a93100e1c8a3a450b
fetch_one \
  Nemotron-Pretraining-Unconditional-Algorithmic/part_000000.parquet \
  unconditional_algorithmic.parquet \
  208343534 \
  706bab9781f6541cb9f40b8af61242e45c6e3dabc0d2f9b3171f08571da37ccb

receipt=$RAW/SHA256SUMS
[[ ! -e "$receipt" && ! -L "$receipt" ]] || {
  echo "[nemotron-v11] source receipt already exists" >&2
  exit 8
}
printf '%s  %s\n' \
  93594fcc5d7e0a85bb8cbc6561b13d9d4c5c9cf4d21e7f81388829e86542e1bf README.md \
  b8b641309dec27e836db6a2003d3045c6ed060c8d6a23e3a93100e1c8a3a450b formal_logic.parquet \
  706bab9781f6541cb9f40b8af61242e45c6e3dabc0d2f9b3171f08571da37ccb unconditional_algorithmic.parquet \
  > "$receipt"
chmod 0444 "$receipt"
(cd "$RAW" && sha256sum -c SHA256SUMS)
echo "[nemotron-v11] pinned profile sources ready; no training admission"
