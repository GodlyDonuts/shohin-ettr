#!/usr/bin/env bash
#SBATCH --job-name=q36-nemotron-ultra-stage
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --no-requeue
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc33,evc34,evc37,evc38,evc43,evc46,evc50
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/artifacts/q36_mtr_d9ff7f7_r1/logs/q36_nemotron_ultra_stage_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/artifacts/q36_mtr_d9ff7f7_r1/logs/q36_nemotron_ultra_stage_%j.err

set -euo pipefail
umask 077

readonly PYTHON=/lustre/fs1/home/sa305415/shohin/envs/product-reasoning-b3a3603-r2/bin/python
readonly REPOSITORY=nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4
readonly REVISION=183968f87ae4cedce3039313cac1fd43d112c578
readonly FINAL_ROOT=/lustre/fs1/home/sa305415/shohin/artifacts/external/nemotron-3-ultra-550b-a55b-nvfp4-183968f
readonly STAGE_ROOT=/lustre/fs1/home/sa305415/shohin/artifacts/external/.nemotron-3-ultra-550b-a55b-nvfp4-183968f.stage
readonly EXPECTED_FILES=243
readonly EXPECTED_BYTES=352381245521
readonly EXPECTED_CONFIG_SHA256=0c939f324c8910f5ebdafbe2a56d7e4e074c50042a3b4f26326bf71a3fe33929
readonly EXPECTED_INDEX_SHA256=41951cc6db98b717cf45c73dd6f0503c050e99c70a9662abe01d9511bc2bed06
readonly EXPECTED_TOKENIZER_SHA256=623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7
readonly EXPECTED_QUANT_CONFIG_SHA256=68f0c66d927e5e9115ad475a7c1dbfe40f7e7e72d9ab9c3b6a33d8c52da2c596
readonly EXPECTED_TOKENIZER_CONFIG_SHA256=10f93eabcb9b1602fbb991d6308e787ce1df28ee9cd7a1c6d1e8c3f338b957bc
readonly EXPECTED_GENERATION_CONFIG_SHA256=683e2b96f880913a788b185809f32235e19964ba9d5cb414c220b172be9bb8bd
readonly EXPECTED_SOURCE_REVISION_SHA256=72b9c31cfe862b4bb903d0ba7da4452df43875fdf32cd2c388ca836164338c8c
readonly SAFETY_BYTES=68719476736
readonly REQUIRED_HEADROOM_BYTES=421100722257
readonly REQUIRED_HEADROOM_INODES=20000

[[ -x "$PYTHON" ]]
[[ ! -e "$FINAL_ROOT" && ! -e "$STAGE_ROOT" ]]
[[ -d "$(dirname "$FINAL_ROOT")" ]]

# Refuse to create even a staging directory unless the durable filesystem can
# hold the complete exact snapshot and retain 64 GiB of post-download margin.
quota_raw=$(/usr/bin/lfs quota -u "${USER:?}" "$(dirname "$FINAL_ROOT")")
export Q36_ULTRA_QUOTA_RAW=$quota_raw
quota_receipt=$(env -i \
  PATH=/usr/bin:/bin \
  Q36_ULTRA_QUOTA_RAW="$Q36_ULTRA_QUOTA_RAW" \
  PYTHONDONTWRITEBYTECODE=1 \
  "$PYTHON" -P -s -B - \
  "$REQUIRED_HEADROOM_BYTES" "$REQUIRED_HEADROOM_INODES" <<'PY'
import os
import re
import sys

minimum_bytes, minimum_inodes = map(int, sys.argv[1:])
candidates = []
for line in os.environ["Q36_ULTRA_QUOTA_RAW"].splitlines():
    values = [
        int(token.rstrip("*"))
        for token in line.split()
        if re.fullmatch(r"\d+\*?", token)
    ]
    if len(values) >= 6:
        candidates.append(values[:6])
if len(candidates) != 1:
    raise SystemExit("Q36 Ultra quota output geometry differs")
used_kib, _soft_kib, limit_kib, used_inodes, _soft_inodes, limit_inodes = candidates[0]
headroom_bytes = (limit_kib - used_kib) * 1024
headroom_inodes = limit_inodes - used_inodes
if headroom_bytes < minimum_bytes or headroom_inodes < minimum_inodes:
    raise SystemExit("Q36 Ultra durable-storage headroom is unsafe")
print(used_kib, limit_kib, used_inodes, limit_inodes, headroom_bytes, headroom_inodes)
PY
)
read -r USED_KIB LIMIT_KIB USED_INODES LIMIT_INODES HEADROOM_BYTES HEADROOM_INODES <<<"$quota_receipt"
export USED_KIB LIMIT_KIB USED_INODES LIMIT_INODES HEADROOM_BYTES HEADROOM_INODES

cache_root=$(/usr/bin/mktemp -d "/tmp/q36-nemotron-ultra-${SLURM_JOB_ID}.XXXXXX")
cleanup() {
  if [[ -n "${cache_root:-}" && -d "$cache_root" ]]; then
    /usr/bin/rm -rf --one-file-system -- "$cache_root"
  fi
}
trap cleanup EXIT

/usr/bin/mkdir -- "$STAGE_ROOT"
env -i \
  PATH=/usr/bin:/bin \
  HOME="$cache_root/home" \
  HF_HOME="$cache_root/huggingface" \
  HF_HUB_CACHE="$cache_root/huggingface/hub" \
  HF_XET_CACHE="$cache_root/huggingface/xet" \
  PYTHONDONTWRITEBYTECODE=1 \
  "$PYTHON" -P -s -B - "$REPOSITORY" "$REVISION" "$STAGE_ROOT" <<'PY'
from pathlib import Path
import sys

from huggingface_hub import HfApi, snapshot_download

repository, revision, stage = sys.argv[1:]
info = HfApi().model_info(repository, revision=revision, files_metadata=True)
if info.sha != revision:
    raise SystemExit(f"resolved revision differs: {info.sha}")
if len(info.siblings) != 243:
    raise SystemExit("Q36 Ultra upstream file count differs")
if sum(int(value.size or 0) for value in info.siblings) != 352_381_245_521:
    raise SystemExit("Q36 Ultra upstream byte count differs")
snapshot_download(
    repo_id=repository,
    revision=revision,
    local_dir=Path(stage),
    max_workers=16,
)
PY

/usr/bin/rm -rf --one-file-system -- "$STAGE_ROOT/.cache"
[[ -z "$(/usr/bin/find "$STAGE_ROOT" -type l -print -quit)" ]]
[[ -z "$(/usr/bin/find "$STAGE_ROOT" ! -type d ! -type f -print -quit)" ]]
observed_files=$(/usr/bin/find "$STAGE_ROOT" -type f | /usr/bin/wc -l | /usr/bin/tr -d ' ')
observed_bytes=$(/usr/bin/find "$STAGE_ROOT" -type f -printf '%s\n' | /usr/bin/awk '{ total += $1 } END { printf "%.0f", total }')
[[ "$observed_files" == "$EXPECTED_FILES" ]]
[[ "$observed_bytes" == "$EXPECTED_BYTES" ]]

verify_sha256() {
  local path=$1
  local expected=$2
  [[ "$(/usr/bin/sha256sum "$path" | /usr/bin/cut -d' ' -f1)" == "$expected" ]]
}
verify_sha256 "$STAGE_ROOT/config.json" "$EXPECTED_CONFIG_SHA256"
verify_sha256 "$STAGE_ROOT/model.safetensors.index.json" "$EXPECTED_INDEX_SHA256"
verify_sha256 "$STAGE_ROOT/tokenizer.json" "$EXPECTED_TOKENIZER_SHA256"
verify_sha256 "$STAGE_ROOT/hf_quant_config.json" "$EXPECTED_QUANT_CONFIG_SHA256"
verify_sha256 "$STAGE_ROOT/tokenizer_config.json" "$EXPECTED_TOKENIZER_CONFIG_SHA256"
verify_sha256 "$STAGE_ROOT/generation_config.json" "$EXPECTED_GENERATION_CONFIG_SHA256"

/usr/bin/printf '%s\n' "$REVISION" > "$STAGE_ROOT/SOURCE_REVISION"
verify_sha256 "$STAGE_ROOT/SOURCE_REVISION" "$EXPECTED_SOURCE_REVISION_SHA256"

env -i \
  PATH=/usr/bin:/bin \
  PYTHONDONTWRITEBYTECODE=1 \
  USED_KIB="$USED_KIB" \
  LIMIT_KIB="$LIMIT_KIB" \
  USED_INODES="$USED_INODES" \
  LIMIT_INODES="$LIMIT_INODES" \
  HEADROOM_BYTES="$HEADROOM_BYTES" \
  HEADROOM_INODES="$HEADROOM_INODES" \
  "$PYTHON" -P -s -B - \
  "$STAGE_ROOT" "$REPOSITORY" "$REVISION" "$SLURM_JOB_ID" \
  "$EXPECTED_FILES" "$EXPECTED_BYTES" "$SAFETY_BYTES" <<'PY'
from pathlib import Path
import hashlib
import json
import os
import sys

root = Path(sys.argv[1])
payload = {
    "schema": "shohin-upward-moe-model-snapshot-v1",
    "status": "complete",
    "repository": sys.argv[2],
    "revision": sys.argv[3],
    "slurm_job_id": int(sys.argv[4]),
    "upstream_files": int(sys.argv[5]),
    "upstream_bytes": int(sys.argv[6]),
    "post_download_safety_bytes": int(sys.argv[7]),
    "quota_preflight": {
        "used_kib": int(os.environ["USED_KIB"]),
        "limit_kib": int(os.environ["LIMIT_KIB"]),
        "used_inodes": int(os.environ["USED_INODES"]),
        "limit_inodes": int(os.environ["LIMIT_INODES"]),
        "headroom_bytes": int(os.environ["HEADROOM_BYTES"]),
        "headroom_inodes": int(os.environ["HEADROOM_INODES"]),
    },
    "source_revision_sha256": hashlib.sha256((root / "SOURCE_REVISION").read_bytes()).hexdigest(),
}
path = root / "snapshot_receipt.json"
with path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

# The manifest is deliberately built after the receipt so exact-membership
# verification covers every candidate-visible model-tree byte. SHA256SUMS is
# the sole excluded file and its digest is supplied independently downstream.
(
  cd "$STAGE_ROOT"
  /usr/bin/find . -type f ! -name SHA256SUMS -printf '%P\0' \
    | /usr/bin/sort -z \
    | /usr/bin/xargs -0 /usr/bin/sha256sum -- \
    > SHA256SUMS
  [[ "$(/usr/bin/wc -l < SHA256SUMS | /usr/bin/tr -d ' ')" == 245 ]]
  /usr/bin/sha256sum -c SHA256SUMS
)

/usr/bin/chmod -R a-w "$STAGE_ROOT"
/usr/bin/mv -- "$STAGE_ROOT" "$FINAL_ROOT"
[[ -d "$FINAL_ROOT" && ! -e "$STAGE_ROOT" ]]
[[ -z "$(/usr/bin/find "$FINAL_ROOT" -xdev -perm /0222 -print -quit)" ]]
(
  cd "$FINAL_ROOT"
  /usr/bin/sha256sum -c SHA256SUMS
)
/usr/bin/sha256sum "$FINAL_ROOT/SHA256SUMS" "$FINAL_ROOT/snapshot_receipt.json"
