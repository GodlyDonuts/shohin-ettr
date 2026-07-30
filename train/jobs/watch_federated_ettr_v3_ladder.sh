#!/bin/bash
# Wait for one verified ETTR transfer, then run a fail-closed 100/500/2000
# federated learning ladder inside a fixed set of one-H100 reservations.

set -euo pipefail

ALLOCATION_JOB_IDS=${ALLOCATION_JOB_IDS:?set comma-separated running job IDs}
LAUNCHER_ROOT=${LAUNCHER_ROOT:?set the immutable federated launcher root}
CODE_ROOT=${CODE_ROOT:?set the immutable full ETTR source root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set the exact ETTR source commit}
TRANSFER_ROOT=${TRANSFER_ROOT:?set the final verified Newton transfer root}
PROTECTED_CHECKPOINT=${PROTECTED_CHECKPOINT:?set the protected checkpoint}
OUTPUT_PREFIX=${OUTPUT_PREFIX:?set a fresh absolute output prefix}
ARCHITECTURE_SEED=${ARCHITECTURE_SEED:?set the architecture seed}
DATA_SEED=${DATA_SEED:?set the data seed}
TOTAL_UPDATES=${TOTAL_UPDATES:?set the token-normalized total updates}
WARMUP_UPDATES=${WARMUP_UPDATES:?set the token-normalized warmup updates}
POLL_SECONDS=${POLL_SECONDS:-60}
PYTHON_ROOT=${PYTHON_ROOT:-/lustre/fs1/home/sa305415/shohin/miniforge3}

integer_contract="$ARCHITECTURE_SEED:$DATA_SEED:$TOTAL_UPDATES"
integer_contract+=":$WARMUP_UPDATES:$POLL_SECONDS"
case "$integer_contract" in
  *[!0-9:]* | *::* | :* | *:)
    echo "federated ladder integer settings differ" >&2
    exit 2
    ;;
esac
if (( TOTAL_UPDATES < 2000 || WARMUP_UPDATES >= TOTAL_UPDATES \
  || POLL_SECONDS < 5 || POLL_SECONDS > 600 )); then
  echo "federated ladder schedule differs" >&2
  exit 2
fi
if [[ ! "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "federated ladder source identity differs" >&2
  exit 2
fi
for path in \
  "$LAUNCHER_ROOT" \
  "$CODE_ROOT" \
  "$TRANSFER_ROOT" \
  "$PROTECTED_CHECKPOINT" \
  "$OUTPUT_PREFIX" \
  "$PYTHON_ROOT"; do
  if [[ "$path" != /* || "$path" == *$'\n'* || "$path" == *$'\r'* ]]; then
    echo "federated ladder paths must be absolute single-line paths" >&2
    exit 2
  fi
done
if [[ ! -x "$PYTHON_ROOT/bin/python" \
  || ! -r "$LAUNCHER_ROOT/train/jobs/run_federated_ettr_v3_pilot.sh" ]]; then
  echo "federated ladder runtime differs" >&2
  exit 2
fi

IFS=',' read -r -a job_ids <<< "$ALLOCATION_JOB_IDS"
if (( ${#job_ids[@]} < 2 || ${#job_ids[@]} > 20 )); then
  echo "federated ladder world size differs" >&2
  exit 2
fi
declare -A seen_jobs=()
for job in "${job_ids[@]}"; do
  if [[ ! "$job" =~ ^[0-9]+$ || -n "${seen_jobs[$job]:-}" ]]; then
    echo "federated ladder allocation identity differs" >&2
    exit 2
  fi
  seen_jobs[$job]=1
done

release_root="$TRANSFER_ROOT/release"
data_root="$TRANSFER_ROOT/data"
tokenizer="$TRANSFER_ROOT/tokenizer.json"
transfer_receipt="$TRANSFER_ROOT/transfer-receipt.json"

verify_transfer() {
  "$PYTHON_ROOT/bin/python" - \
    "$transfer_receipt" \
    "$release_root/release.json" \
    "$SOURCE_COMMIT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys


receipt_path, release_path = map(Path, sys.argv[1:3])
source_commit = sys.argv[3]
if not receipt_path.is_file() or not release_path.is_file():
    raise SystemExit(1)
receipt = json.loads(receipt_path.read_text(encoding="ascii"))
claimed = receipt.pop("payload_sha256", None)
canonical = json.dumps(
    receipt,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
).encode("ascii")
release_sha256 = hashlib.sha256(release_path.read_bytes()).hexdigest()
if (
    receipt.get("schema") != "shohin-ettr-il-v3-direct-transfer-receipt-v1"
    or receipt.get("status") != "pass"
    or receipt.get("source_commit") != source_commit
    or claimed != hashlib.sha256(canonical).hexdigest()
    or receipt.get("release_file_sha256") != release_sha256
):
    raise SystemExit(2)
print(release_sha256)
PY
}

while true; do
  for job in "${job_ids[@]}"; do
    state=$(squeue -h -j "$job" -o "%T")
    name=$(squeue -h -j "$job" -o "%j")
    if [[ "$state" != RUNNING || "$name" != shohin-1h100-* ]]; then
      echo "federated ladder lost allocation $job" >&2
      exit 3
    fi
  done
  if release_sha256=$(verify_transfer 2>/dev/null); then
    break
  fi
  printf \
    '%s federated_ladder_waiting jobs=%s output=%s\n' \
    "$(date -Iseconds)" \
    "$ALLOCATION_JOB_IDS" \
    "$OUTPUT_PREFIX"
  sleep "$POLL_SECONDS"
done

start=0
resume_checkpoint=
resume_sha256=
for target in 100 500 2000; do
  output="${OUTPUT_PREFIX}_u${target}"
  if [[ -e "$output" || -L "$output" ]]; then
    echo "federated ladder output already exists: $output" >&2
    exit 4
  fi
  resume_env=()
  if (( start > 0 )); then
    previous="${OUTPUT_PREFIX}_u${start}"
    if ! "$PYTHON_ROOT/bin/python" - \
      "$previous/development-evaluation.json" <<'PY'
import json
from pathlib import Path
import sys


report = json.loads(Path(sys.argv[1]).read_text(encoding="ascii"))
if report.get("gates", {}).get("strict_learning_signal") is not True:
    raise SystemExit(1)
PY
    then
      printf \
        '%s federated_ladder_stopped target=%s reason=strict_gate_failed\n' \
        "$(date -Iseconds)" \
        "$start"
      exit 0
    fi
    resume_checkpoint=$(printf \
      '%s/train/checkpoint-update-%07d.pt' \
      "$previous" \
      "$start")
    resume_sidecar=${resume_checkpoint%.pt}.json
    resume_sha256=$(
      "$PYTHON_ROOT/bin/python" -c \
        'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["checkpoint_sha256"])' \
        "$resume_sidecar"
    )
    if [[ ! -s "$resume_checkpoint" \
      || ! "$resume_sha256" =~ ^[0-9a-f]{64}$ \
      || "$(sha256sum "$resume_checkpoint" | awk '{print $1}')" \
        != "$resume_sha256" ]]; then
      echo "federated ladder resume identity differs" >&2
      exit 5
    fi
    resume_env=(
      RESUME_CHECKPOINT="$resume_checkpoint"
      RESUME_SHA256="$resume_sha256"
    )
  fi

  env \
    ALLOCATION_JOB_IDS="$ALLOCATION_JOB_IDS" \
    CODE_ROOT="$CODE_ROOT" \
    SOURCE_COMMIT="$SOURCE_COMMIT" \
    RELEASE_ROOT="$release_root" \
    RELEASE_SHA256="$release_sha256" \
    DATA_ROOT="$data_root" \
    TOKENIZER="$tokenizer" \
    PROTECTED_CHECKPOINT="$PROTECTED_CHECKPOINT" \
    OUTDIR="$output" \
    START_UPDATE="$start" \
    TARGET_UPDATE="$target" \
    ACCUMULATION=1 \
    CHECKPOINT_EVERY="$((target - start))" \
    LOG_EVERY=10 \
    MAX_EVAL_BATCHES=64 \
    ARCHITECTURE_SEED="$ARCHITECTURE_SEED" \
    DATA_SEED="$DATA_SEED" \
    TOTAL_UPDATES="$TOTAL_UPDATES" \
    WARMUP_UPDATES="$WARMUP_UPDATES" \
    FREEZE_BASE=1 \
    COMPILE_MODE=default \
    LAUNCH_STAGGER_SECONDS=1 \
    PYTHON_ROOT="$PYTHON_ROOT" \
    "${resume_env[@]}" \
    bash "$LAUNCHER_ROOT/train/jobs/run_federated_ettr_v3_pilot.sh"
  start="$target"
done

printf \
  '%s federated_ladder_complete jobs=%s output=%s\n' \
  "$(date -Iseconds)" \
  "$ALLOCATION_JOB_IDS" \
  "$OUTPUT_PREFIX"
