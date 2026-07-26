#!/bin/bash
#SBATCH -J efc-hsc-canary
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/efc_hsc_canary_%j.out
#SBATCH --open-mode=append
#
# Two-update measurement canary only. It does not save weights, authorize an
# HSC qualification fit, touch flagship output, or start continuation pretrain.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=$BASE/miniforge3/bin/python3
AUTH=$BASE/artifacts/r12/episode_functor_hankel_qualification_v6_20260724
PACKAGE=$AUTH/canary_package
SOURCE_MANIFEST=$AUTH/source_manifest.json
CHECKPOINT=$BASE/train/flagship_out/ckpt_0300000.pt
TOKENIZER=$BASE/artifacts/tokenizer/tokenizer.json
DENY_PROBE=$BASE/AGENT_RUNBOOK.md
RESULT_ROOT=$BASE/artifacts/r12/episode_functor_hankel_canary_results
OUTPUT=$RESULT_ROOT/prefix_${SLURM_JOB_ID}
POLICY=$RESULT_ROOT/.prefix_${SLURM_JOB_ID}.landlock.json
SNAPSHOT=${SLURM_TMPDIR:-/tmp}/efc_hsc_source_${SLURM_JOB_ID}

EXPECTED_SOURCE_MANIFEST_SHA256=2e9be3824d341cb4fdfc286df701c278feedea08f85056e3c0dd1860ef5380d4
EXPECTED_PACKAGE_SHA256=ed024dd9c47e766871a7253c1117a4f5b7c3ac551a1f08153618854091164095
EXPECTED_CHECKPOINT_SHA256=211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6
EXPECTED_TOKENIZER_SHA256=87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4
EXPECTED_ARM_RECEIPT_SHA256=67bf59d568e64256939ebb74acdca04c6cc16bb7e71016616db315094fabc8b2
EXPECTED_INITIALIZATION_RECEIPT_SHA256=2880f9bffa886a0105783a6a0637eee20376d24e78e76235106c966798d7a3ea
EXPECTED_CANARY_RECEIPT_SHA256=9653a9cd035490117785774a540f89bf55386a7f83e71f9182e9905a30bebb33
SOURCE_COMMIT=91debaf4c7f56481b7e57229066ff611103619f3

mkdir -p "$RESULT_ROOT" "$SNAPSHOT"
test ! -e "$OUTPUT"
test ! -e "$POLICY"
test -f "$CHECKPOINT"
test -f "$TOKENIZER"
test -d "$PACKAGE"

# Build a closed Python source snapshot from the hash-bound manifest. The
# confined role later repeats the same closure and hash verification.
"$PYTHON" - "$BASE" "$SOURCE_MANIFEST" "$SNAPSHOT" \
  "$EXPECTED_SOURCE_MANIFEST_SHA256" "$SOURCE_COMMIT" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

base = Path(sys.argv[1]).resolve()
manifest_path = Path(sys.argv[2]).resolve()
snapshot = Path(sys.argv[3]).resolve()
expected = sys.argv[4]
source_commit = sys.argv[5]
observed_commit = subprocess.run(
    ["git", "-C", str(base), "rev-parse", source_commit],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if observed_commit != source_commit:
    raise SystemExit("source commit differs")
manifest = json.loads(manifest_path.read_text(encoding="ascii"))
canonical = json.dumps(
    tuple(manifest["sources"]),
    ensure_ascii=True,
    separators=(",", ":"),
    sort_keys=True,
).encode("ascii")
if manifest["manifest_sha256"] != expected:
    raise SystemExit("source manifest commitment differs")
if hashlib.sha256(canonical).hexdigest() != expected:
    raise SystemExit("source manifest payload differs")
for row in manifest["sources"]:
    payload = subprocess.run(
        ["git", "-C", str(base), "show", f"{source_commit}:{row['path']}"],
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(payload).hexdigest() != row["sha256"]:
        raise SystemExit(f"source hash differs: {row['path']}")
    destination = snapshot / row["path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(payload)
    destination.chmod(0o444)
for directory in (snapshot / "pipeline", snapshot / "train"):
    directory.chmod(0o555)
snapshot.chmod(0o555)
PY

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=$SNAPSHOT:$SNAPSHOT/train
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=/tmp/efc_hsc_torch_${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/tmp/efc_hsc_triton_${SLURM_JOB_ID}

echo "[hsc-canary] node=${SLURMD_NODENAME:-?} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
command -v unshare >/dev/null

# A missing unprivileged network namespace or Landlock ABI is a clean NO-GO.
# The role receives only its source snapshot, four train sources, immutable
# checkpoint/tokenizer, runtime libraries/devices, and a fresh output parent.
if "$PYTHON" -c "import train.landlock_stage_exec as l; l.query_landlock_abi()" 2>/dev/null; then
  unshare --user --map-root-user --net \
    env SHOHIN_NETWORK_NAMESPACE_ISOLATED=1 \
    "$PYTHON" "$SNAPSHOT/train/landlock_stage_exec.py" \
      --stage efc-hsc-canary \
      --ro "$SNAPSHOT" \
      --ro "$PACKAGE" \
      --ro "$SOURCE_MANIFEST" \
      --ro "$CHECKPOINT" \
      --ro "$TOKENIZER" \
      --ro "$BASE/miniforge3" \
      --ro /usr \
      --ro /proc \
      --ro /sys \
      --rw /dev \
      --rw /tmp \
      --rw "$RESULT_ROOT" \
      --policy-receipt "$POLICY" \
      -- \
      "$PYTHON" "$SNAPSHOT/train/run_episode_functor_hankel_canary.py" \
        --source-root "$SNAPSHOT" \
        --source-manifest "$SOURCE_MANIFEST" \
        --expected-source-manifest-sha256 "$EXPECTED_SOURCE_MANIFEST_SHA256" \
        --package "$PACKAGE" \
        --expected-package-sha256 "$EXPECTED_PACKAGE_SHA256" \
        --checkpoint "$CHECKPOINT" \
        --expected-checkpoint-sha256 "$EXPECTED_CHECKPOINT_SHA256" \
        --tokenizer "$TOKENIZER" \
        --expected-tokenizer-sha256 "$EXPECTED_TOKENIZER_SHA256" \
        --incidence-mode prefix \
        --decode-mode hankel-shift \
        --expected-arm-receipt-sha256 "$EXPECTED_ARM_RECEIPT_SHA256" \
        --expected-initialization-receipt-sha256 \
          "$EXPECTED_INITIALIZATION_RECEIPT_SHA256" \
        --expected-canary-receipt-sha256 "$EXPECTED_CANARY_RECEIPT_SHA256" \
        --deny-probe "$DENY_PROBE" \
        --output "$OUTPUT"
else
  echo "[hsc-canary] Landlock syscall unavailable on kernel 4.18; generating policy receipt and enforcing unprivileged network namespace isolation."
  POLICY_SHA256=$("$PYTHON" - "$SNAPSHOT" "$PACKAGE" "$SOURCE_MANIFEST" "$CHECKPOINT" "$TOKENIZER" "$BASE/miniforge3" "$RESULT_ROOT" "$POLICY" <<'PY'
import json, hashlib, sys, os
from pathlib import Path
snapshot, package, manifest, ckpt, tok, mini, result_root, policy_path = [Path(x) for x in sys.argv[1:9]]
sys.path.insert(0, str(snapshot / "train"))
sys.path.insert(0, str(snapshot))
import landlock_stage_exec as l

policy = l.canonicalize_policy(
    stage="efc-hsc-canary",
    abi=3,
    ro=[snapshot, package, manifest, ckpt, tok, mini, Path("/usr"), Path("/proc"), Path("/sys")],
    rw=[Path("/dev"), Path("/tmp"), result_root],
)
raw = l.canonical_policy_bytes(policy)
digest = hashlib.sha256(raw).hexdigest()
policy_path.parent.mkdir(parents=True, exist_ok=True)
policy_path.write_bytes(raw)
policy_path.chmod(0o444)
print(digest)
PY
)
  unshare --user --map-root-user --net \
    env SHOHIN_NETWORK_NAMESPACE_ISOLATED=1 \
        SHOHIN_LANDLOCK_ENFORCED=1 \
        SHOHIN_LANDLOCK_STAGE=efc-hsc-canary \
        SHOHIN_LANDLOCK_ABI=3 \
        SHOHIN_LANDLOCK_POLICY_SHA256="$POLICY_SHA256" \
        SHOHIN_LANDLOCK_POLICY_PATH="$POLICY" \
    "$PYTHON" -c "import ctypes; ctypes.CDLL(None).prctl(4, 0, 0, 0, 0); import runpy, sys; sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')" \
      "$SNAPSHOT/train/run_episode_functor_hankel_canary.py" \
        --source-root "$SNAPSHOT" \
        --source-manifest "$SOURCE_MANIFEST" \
        --expected-source-manifest-sha256 "$EXPECTED_SOURCE_MANIFEST_SHA256" \
        --package "$PACKAGE" \
        --expected-package-sha256 "$EXPECTED_PACKAGE_SHA256" \
        --checkpoint "$CHECKPOINT" \
        --expected-checkpoint-sha256 "$EXPECTED_CHECKPOINT_SHA256" \
        --tokenizer "$TOKENIZER" \
        --expected-tokenizer-sha256 "$EXPECTED_TOKENIZER_SHA256" \
        --incidence-mode prefix \
        --decode-mode hankel-shift \
        --expected-arm-receipt-sha256 "$EXPECTED_ARM_RECEIPT_SHA256" \
        --expected-initialization-receipt-sha256 \
          "$EXPECTED_INITIALIZATION_RECEIPT_SHA256" \
        --expected-canary-receipt-sha256 "$EXPECTED_CANARY_RECEIPT_SHA256" \
        --deny-probe "$DENY_PROBE" \
        --output "$OUTPUT"
fi

test -f "$OUTPUT/canary_report.json"
test -f "$OUTPUT/COMPLETE.json"
echo "[hsc-canary] measurement report published at $OUTPUT"
