#!/usr/bin/env bash
#SBATCH --job-name=q36-nemotron-ultra-restore
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --no-requeue
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc33,evc34,evc37,evc38,evc43,evc46,evc50
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/artifacts/q36_mtr_d9ff7f7_r1/logs/q36_nemotron_ultra_restore_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/artifacts/q36_mtr_d9ff7f7_r1/logs/q36_nemotron_ultra_restore_%j.err

set -euo pipefail
umask 077

readonly PYTHON=/lustre/fs1/home/sa305415/shohin/envs/product-reasoning-b3a3603-r2/bin/python
readonly REPOSITORY=nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4
readonly REVISION=183968f87ae4cedce3039313cac1fd43d112c578
readonly EVIDENCE_ROOT=/lustre/fs1/home/sa305415/shohin/artifacts/q36_ultra_snapshot_evidence_183968f_r1
readonly FINAL_ROOT=/lustre/fs1/home/sa305415/shohin/artifacts/external/nemotron-3-ultra-550b-a55b-nvfp4-183968f
readonly STAGE_ROOT=/lustre/fs1/home/sa305415/shohin/artifacts/external/.nemotron-3-ultra-550b-a55b-nvfp4-183968f.restore-stage
readonly RESTORE_ROOT=/lustre/fs1/home/sa305415/shohin/artifacts/q36_ultra_restore_183968f_r1
readonly RESTORE_STAGE=/lustre/fs1/home/sa305415/shohin/artifacts/.q36_ultra_restore_183968f_r1.stage
readonly EXPECTED_UPSTREAM_FILES=243
readonly EXPECTED_UPSTREAM_BYTES=352381245521
readonly EXPECTED_MANIFEST_ENTRIES=245
readonly EXPECTED_WEIGHT_FILES=113
readonly EXPECTED_NONWEIGHT_FILES=132
readonly EXPECTED_FINAL_FILES=246
readonly EXPECTED_CAPSULE_FILES=134
readonly EXPECTED_CAPSULE_MANIFEST_ENTRIES=133
readonly EXPECTED_CAPSULE_MANIFEST_SHA256=4f2b30d3c5090f81765b32d8afc51c9d6379999379a5247c8d9497fe9cb68641
readonly EXPECTED_MODEL_MANIFEST_SHA256=e20d8aad607dd1146ee83d49c31f65e7fc24033348137a57a93615436484751a
readonly EXPECTED_ORIGINAL_RECEIPT_SHA256=6b9b50ed6bdb644c0bf059c3c9f8cff2a2775d936fb8c6d1b65f0c953aba731a
readonly EXPECTED_SOURCE_REVISION_SHA256=72b9c31cfe862b4bb903d0ba7da4452df43875fdf32cd2c388ca836164338c8c
readonly POST_RESTORE_SAFETY_BYTES=68719476736
readonly REQUIRED_HEADROOM_BYTES=421100722257
readonly REQUIRED_HEADROOM_INODES=20000

[[ -x "$PYTHON" ]]
[[ -d "$EVIDENCE_ROOT" && ! -L "$EVIDENCE_ROOT" ]]
[[ ! -e "$FINAL_ROOT" && ! -e "$STAGE_ROOT" ]]
[[ ! -e "$RESTORE_ROOT" && ! -e "$RESTORE_STAGE" ]]
[[ -d "$(dirname "$FINAL_ROOT")" && -d "$(dirname "$RESTORE_ROOT")" ]]

verify_sha256() {
  local path=$1
  local expected=$2
  [[ -f "$path" && ! -L "$path" ]]
  [[ "$(/usr/bin/sha256sum "$path" | /usr/bin/cut -d' ' -f1)" == "$expected" ]]
}

# The capsule is the immutable custody authority. It contains the original
# complete model manifest and every non-weight file, so restoration needs to
# fetch only the 113 safetensor shards and can prove byte-for-byte identity.
[[ -z "$(/usr/bin/find "$EVIDENCE_ROOT" -type l -print -quit)" ]]
[[ -z "$(/usr/bin/find "$EVIDENCE_ROOT" ! -type d ! -type f -print -quit)" ]]
[[ -z "$(/usr/bin/find "$EVIDENCE_ROOT" -xdev -perm /0222 -print -quit)" ]]
[[ "$(/usr/bin/find "$EVIDENCE_ROOT" -type f | /usr/bin/wc -l | /usr/bin/tr -d ' ')" == "$EXPECTED_CAPSULE_FILES" ]]
verify_sha256 "$EVIDENCE_ROOT/SHA256SUMS.capsule" "$EXPECTED_CAPSULE_MANIFEST_SHA256"
verify_sha256 "$EVIDENCE_ROOT/SHA256SUMS" "$EXPECTED_MODEL_MANIFEST_SHA256"
verify_sha256 "$EVIDENCE_ROOT/snapshot_receipt.json" "$EXPECTED_ORIGINAL_RECEIPT_SHA256"
verify_sha256 "$EVIDENCE_ROOT/SOURCE_REVISION" "$EXPECTED_SOURCE_REVISION_SHA256"
(
  cd "$EVIDENCE_ROOT"
  [[ "$(/usr/bin/wc -l < SHA256SUMS.capsule | /usr/bin/tr -d ' ')" == "$EXPECTED_CAPSULE_MANIFEST_ENTRIES" ]]
  /usr/bin/sha256sum -c SHA256SUMS.capsule
)

# Validate the preserved manifest grammar and capsule projection before any
# durable mutation. Every path is a unique root-level regular filename.
env -i \
  PATH=/usr/bin:/bin \
  PYTHONDONTWRITEBYTECODE=1 \
  "$PYTHON" -P -s -B - \
  "$EVIDENCE_ROOT" "$EXPECTED_MANIFEST_ENTRIES" "$EXPECTED_WEIGHT_FILES" \
  "$EXPECTED_NONWEIGHT_FILES" <<'PY'
from pathlib import Path
import hashlib
import re
import sys

root = Path(sys.argv[1])
expected_entries, expected_weights, expected_nonweights = map(int, sys.argv[2:])
rows = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
if len(rows) != expected_entries:
    raise SystemExit("Q36 Ultra preserved manifest entry count differs")
entries: dict[str, str] = {}
for row in rows:
    match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", row)
    if match is None:
        raise SystemExit("Q36 Ultra preserved manifest grammar differs")
    digest, name = match.groups()
    if name in entries or name in {".", "..", "SHA256SUMS", "SHA256SUMS.capsule"}:
        raise SystemExit("Q36 Ultra preserved manifest membership differs")
    entries[name] = digest
weights = sorted(name for name in entries if name.endswith(".safetensors"))
nonweights = sorted(name for name in entries if not name.endswith(".safetensors"))
if len(weights) != expected_weights or len(nonweights) != expected_nonweights:
    raise SystemExit("Q36 Ultra preserved manifest projection differs")
for name in nonweights:
    path = root / name
    if not path.is_file() or path.is_symlink():
        raise SystemExit("Q36 Ultra preserved non-weight file is absent")
    if hashlib.sha256(path.read_bytes()).hexdigest() != entries[name]:
        raise SystemExit("Q36 Ultra preserved non-weight digest differs")
for name in weights:
    if (root / name).exists():
        raise SystemExit("Q36 Ultra capsule unexpectedly contains a weight shard")
PY

# Refuse to create a staging directory unless the durable filesystem can hold
# the complete exact snapshot and retain 64 GiB of margin afterward.
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

cache_root=$(/usr/bin/mktemp -d "/tmp/q36-nemotron-ultra-restore-${SLURM_JOB_ID}.XXXXXX")
cleanup() {
  if [[ -n "${cache_root:-}" && -d "$cache_root" ]]; then
    /usr/bin/rm -rf --one-file-system -- "$cache_root"
  fi
}
trap cleanup EXIT

/usr/bin/mkdir -- "$STAGE_ROOT" "$RESTORE_STAGE"

# Project only the original manifest and its 132 exact non-weight members into
# the new tree. The old acquisition receipt intentionally remains in the model
# root so the preserved 245-entry manifest can be replayed without exceptions.
env -i \
  PATH=/usr/bin:/bin \
  PYTHONDONTWRITEBYTECODE=1 \
  "$PYTHON" -P -s -B - "$EVIDENCE_ROOT" "$STAGE_ROOT" <<'PY'
from pathlib import Path
import re
import shutil
import sys

source, destination = map(Path, sys.argv[1:])
rows = (source / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
names = []
for row in rows:
    match = re.fullmatch(r"[0-9a-f]{64}  ([A-Za-z0-9_.-]+)", row)
    if match is None:
        raise SystemExit("Q36 Ultra manifest changed after preflight")
    name = match.group(1)
    if not name.endswith(".safetensors"):
        names.append(name)
for name in names:
    shutil.copyfile(source / name, destination / name)
shutil.copyfile(source / "SHA256SUMS", destination / "SHA256SUMS")
PY

# Query the immutable upstream revision and fetch only the missing weight
# shards. Metadata, tokenizer, code, config, and the original receipt are never
# redownloaded; their authoritative bytes come from the preserved capsule.
env -i \
  PATH=/usr/bin:/bin \
  HOME="$cache_root/home" \
  HF_HOME="$cache_root/huggingface" \
  HF_HUB_CACHE="$cache_root/huggingface/hub" \
  HF_XET_CACHE="$cache_root/huggingface/xet" \
  PYTHONDONTWRITEBYTECODE=1 \
  "$PYTHON" -P -s -B - \
  "$REPOSITORY" "$REVISION" "$STAGE_ROOT" \
  "$EXPECTED_UPSTREAM_FILES" "$EXPECTED_UPSTREAM_BYTES" "$EXPECTED_WEIGHT_FILES" <<'PY'
from pathlib import Path
import sys

from huggingface_hub import HfApi, snapshot_download

repository, revision, stage = sys.argv[1:4]
expected_files, expected_bytes, expected_weights = map(int, sys.argv[4:])
info = HfApi().model_info(repository, revision=revision, files_metadata=True)
if info.sha != revision:
    raise SystemExit(f"resolved revision differs: {info.sha}")
if len(info.siblings) != expected_files:
    raise SystemExit("Q36 Ultra upstream file count differs")
if sum(int(value.size or 0) for value in info.siblings) != expected_bytes:
    raise SystemExit("Q36 Ultra upstream byte count differs")
weights = sorted(value.rfilename for value in info.siblings if value.rfilename.endswith(".safetensors"))
if len(weights) != expected_weights or any("/" in value for value in weights):
    raise SystemExit("Q36 Ultra upstream weight projection differs")
snapshot_download(
    repo_id=repository,
    revision=revision,
    local_dir=Path(stage),
    allow_patterns=weights,
    max_workers=16,
)
PY

/usr/bin/rm -rf --one-file-system -- "$STAGE_ROOT/.cache"
[[ -z "$(/usr/bin/find "$STAGE_ROOT" -type l -print -quit)" ]]
[[ -z "$(/usr/bin/find "$STAGE_ROOT" ! -type d ! -type f -print -quit)" ]]
[[ "$(/usr/bin/find "$STAGE_ROOT" -type f | /usr/bin/wc -l | /usr/bin/tr -d ' ')" == "$EXPECTED_FINAL_FILES" ]]
verify_sha256 "$STAGE_ROOT/SHA256SUMS" "$EXPECTED_MODEL_MANIFEST_SHA256"
(
  cd "$STAGE_ROOT"
  /usr/bin/sha256sum -c SHA256SUMS
)

# Reject extra or missing files independently of sha256sum -c, which validates
# listed bytes but does not enforce exact tree membership.
env -i \
  PATH=/usr/bin:/bin \
  PYTHONDONTWRITEBYTECODE=1 \
  "$PYTHON" -P -s -B - "$STAGE_ROOT" "$EXPECTED_MANIFEST_ENTRIES" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
expected_entries = int(sys.argv[2])
rows = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
names = []
for row in rows:
    match = re.fullmatch(r"[0-9a-f]{64}  ([A-Za-z0-9_.-]+)", row)
    if match is None:
        raise SystemExit("Q36 Ultra restored manifest grammar differs")
    names.append(match.group(1))
if len(names) != expected_entries or len(set(names)) != expected_entries:
    raise SystemExit("Q36 Ultra restored manifest membership differs")
observed = sorted(path.name for path in root.iterdir() if path.is_file())
if observed != sorted([*names, "SHA256SUMS"]):
    raise SystemExit("Q36 Ultra restored exact file membership differs")
if any(path.is_dir() for path in root.iterdir()):
    raise SystemExit("Q36 Ultra restored tree unexpectedly contains directories")
PY

# The restoration receipt lives outside the byte-identical model tree and binds
# the new job/quota evidence to the unchanged original manifest and receipt.
env -i \
  PATH=/usr/bin:/bin \
  PYTHONDONTWRITEBYTECODE=1 \
  USED_KIB="$USED_KIB" LIMIT_KIB="$LIMIT_KIB" \
  USED_INODES="$USED_INODES" LIMIT_INODES="$LIMIT_INODES" \
  HEADROOM_BYTES="$HEADROOM_BYTES" HEADROOM_INODES="$HEADROOM_INODES" \
  "$PYTHON" -P -s -B - \
  "$RESTORE_STAGE/restore_receipt.json" "$FINAL_ROOT" "$EVIDENCE_ROOT" \
  "$REPOSITORY" "$REVISION" "$SLURM_JOB_ID" \
  "$EXPECTED_MODEL_MANIFEST_SHA256" "$EXPECTED_ORIGINAL_RECEIPT_SHA256" \
  "$EXPECTED_CAPSULE_MANIFEST_SHA256" "$POST_RESTORE_SAFETY_BYTES" <<'PY'
from pathlib import Path
import json
import os
import sys

(
    output,
    final_root,
    evidence_root,
    repository,
    revision,
    job_id,
    manifest_sha256,
    original_receipt_sha256,
    capsule_manifest_sha256,
    safety_bytes,
) = sys.argv[1:]
payload = {
    "schema": "shohin-q36-nemotron-ultra-exact-restore-v1",
    "status": "complete",
    "repository": repository,
    "revision": revision,
    "slurm_job_id": int(job_id),
    "restored_model_root": final_root,
    "evidence_root": evidence_root,
    "model_manifest_sha256": manifest_sha256,
    "original_snapshot_receipt_sha256": original_receipt_sha256,
    "capsule_manifest_sha256": capsule_manifest_sha256,
    "weight_files_downloaded": 113,
    "nonweight_files_downloaded": 0,
    "byte_exact_original_tree": True,
    "post_restore_safety_bytes": int(safety_bytes),
    "quota_preflight": {
        "used_kib": int(os.environ["USED_KIB"]),
        "limit_kib": int(os.environ["LIMIT_KIB"]),
        "used_inodes": int(os.environ["USED_INODES"]),
        "limit_inodes": int(os.environ["LIMIT_INODES"]),
        "headroom_bytes": int(os.environ["HEADROOM_BYTES"]),
        "headroom_inodes": int(os.environ["HEADROOM_INODES"]),
    },
}
path = Path(output)
with path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

/usr/bin/chmod -R a-w "$STAGE_ROOT" "$RESTORE_STAGE"
/usr/bin/mv -- "$STAGE_ROOT" "$FINAL_ROOT"
/usr/bin/mv -- "$RESTORE_STAGE" "$RESTORE_ROOT"
[[ -d "$FINAL_ROOT" && -d "$RESTORE_ROOT" ]]
[[ ! -e "$STAGE_ROOT" && ! -e "$RESTORE_STAGE" ]]
[[ -z "$(/usr/bin/find "$FINAL_ROOT" "$RESTORE_ROOT" -xdev -perm /0222 -print -quit)" ]]
verify_sha256 "$FINAL_ROOT/SHA256SUMS" "$EXPECTED_MODEL_MANIFEST_SHA256"
(
  cd "$FINAL_ROOT"
  /usr/bin/sha256sum -c SHA256SUMS
)
/usr/bin/sha256sum "$FINAL_ROOT/SHA256SUMS" "$FINAL_ROOT/snapshot_receipt.json" "$RESTORE_ROOT/restore_receipt.json"
