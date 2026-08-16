#!/usr/bin/env bash
#SBATCH --job-name=q36-nemotron-super-stage
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --no-requeue
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc33,evc34,evc37,evc38,evc43,evc46,evc50
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/artifacts/q36_mtr_d9ff7f7_r1/logs/q36_nemotron_super_stage_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/artifacts/q36_mtr_d9ff7f7_r1/logs/q36_nemotron_super_stage_%j.err

set -euo pipefail
umask 077

readonly PYTHON=/lustre/fs1/home/sa305415/shohin/envs/product-reasoning-b3a3603-r2/bin/python
readonly REPOSITORY=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8
readonly REVISION=7d7e5797b8a3c7abbab54033b6004e93e8b6bc91
readonly FINAL_ROOT=/lustre/fs1/home/sa305415/shohin/artifacts/external/nemotron-3-super-120b-a12b-fp8-7d7e5797
readonly STAGE_ROOT=/lustre/fs1/home/sa305415/shohin/artifacts/external/.nemotron-3-super-120b-a12b-fp8-7d7e5797.stage
readonly EXPECTED_FILES=45
readonly EXPECTED_BYTES=128379948583
readonly EXPECTED_CONFIG_SHA256=ff5d6d643b288d4149b0bf820ecb5fe87dd9bbc08b6b811241c57840e11e30e3
readonly EXPECTED_MODELING_SHA256=e1cb5fc02e887983f0a445bf4c1a2604453b2cb2db4624c7004dcf663bbb1b6e
readonly EXPECTED_INDEX_SHA256=126f3105feb375f4f0390aa2b339d5d27d37d1dc720f797bafb4263252c12628

[[ -x "$PYTHON" ]]
[[ ! -e "$FINAL_ROOT" && ! -e "$STAGE_ROOT" ]]
[[ -d "$(dirname "$FINAL_ROOT")" ]]

cache_root=$(/usr/bin/mktemp -d "/tmp/q36-nemotron-super-${SLURM_JOB_ID}.XXXXXX")
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
observed = HfApi().model_info(repository, revision=revision).sha
if observed != revision:
    raise SystemExit(f"resolved revision differs: {observed}")
snapshot_download(
    repo_id=repository,
    revision=revision,
    local_dir=Path(stage),
    max_workers=8,
)
PY

/usr/bin/rm -rf --one-file-system -- "$STAGE_ROOT/.cache"
[[ -z "$(/usr/bin/find "$STAGE_ROOT" -type l -print -quit)" ]]
observed_files=$(/usr/bin/find "$STAGE_ROOT" -type f | /usr/bin/wc -l | /usr/bin/tr -d ' ')
observed_bytes=$(/usr/bin/find "$STAGE_ROOT" -type f -printf '%s\n' | /usr/bin/awk '{ total += $1 } END { printf "%.0f", total }')
[[ "$observed_files" == "$EXPECTED_FILES" ]]
[[ "$observed_bytes" == "$EXPECTED_BYTES" ]]

[[ "$(/usr/bin/sha256sum "$STAGE_ROOT/config.json" | /usr/bin/cut -d' ' -f1)" == "$EXPECTED_CONFIG_SHA256" ]]
[[ "$(/usr/bin/sha256sum "$STAGE_ROOT/modeling_nemotron_h.py" | /usr/bin/cut -d' ' -f1)" == "$EXPECTED_MODELING_SHA256" ]]
[[ "$(/usr/bin/sha256sum "$STAGE_ROOT/model.safetensors.index.json" | /usr/bin/cut -d' ' -f1)" == "$EXPECTED_INDEX_SHA256" ]]

/usr/bin/printf '%s\n' "$REVISION" > "$STAGE_ROOT/SOURCE_REVISION"
(
  cd "$STAGE_ROOT"
  /usr/bin/find . -type f ! -name SHA256SUMS ! -name snapshot_receipt.json -printf '%P\0' \
    | /usr/bin/sort -z \
    | /usr/bin/xargs -0 /usr/bin/sha256sum -- \
    > SHA256SUMS
  /usr/bin/sha256sum -c SHA256SUMS
)

env -i PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -P -s -B - \
  "$STAGE_ROOT" "$REPOSITORY" "$REVISION" "$SLURM_JOB_ID" \
  "$EXPECTED_FILES" "$EXPECTED_BYTES" <<'PY'
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
    "sha256sums_sha256": hashlib.sha256((root / "SHA256SUMS").read_bytes()).hexdigest(),
    "source_revision_sha256": hashlib.sha256((root / "SOURCE_REVISION").read_bytes()).hexdigest(),
}
path = root / "snapshot_receipt.json"
with path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

/usr/bin/chmod -R a-w "$STAGE_ROOT"
/usr/bin/mv -- "$STAGE_ROOT" "$FINAL_ROOT"
[[ -d "$FINAL_ROOT" && ! -e "$STAGE_ROOT" ]]
[[ -z "$(/usr/bin/find "$FINAL_ROOT" -xdev -perm /0222 -print -quit)" ]]
/usr/bin/sha256sum "$FINAL_ROOT/SHA256SUMS" "$FINAL_ROOT/snapshot_receipt.json"
