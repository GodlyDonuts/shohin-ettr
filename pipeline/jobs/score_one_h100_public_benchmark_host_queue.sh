#!/bin/bash
set -euo pipefail

: "${ALLOCATION_JOB_ID:?}" "${RUNTIME_ROOT:?}" "${ARTIFACT_ROOT:?}"
: "${BOARD_ROOT:?}" "${PYTHON:?}"

POLL_SECONDS=${POLL_SECONDS:-30}
CLAIM="$ARTIFACT_ROOT/official_score_controller.json"
SCORER="$RUNTIME_ROOT/pipeline/jobs/score_one_h100_public_benchmark_queue.sh"

if [ "$POLL_SECONDS" -le 0 ]; then
  echo "poll interval must be positive" >&2
  exit 2
fi
for path in "$RUNTIME_ROOT" "$ARTIFACT_ROOT" "$BOARD_ROOT" "$SCORER"; do
  [ -e "$path" ] && [ ! -L "$path" ] || {
    echo "required score-controller input differs: $path" >&2
    exit 2
  }
done

source_commit=$(git -C "$RUNTIME_ROOT" rev-parse HEAD)
if ! runtime_status=$(git -c core.preloadIndex=false -C "$RUNTIME_ROOT" \
  status --porcelain --untracked-files=all); then
  echo "score-controller runtime status check failed" >&2
  exit 2
fi
if [ -n "$runtime_status" ]; then
  echo "score-controller runtime is dirty" >&2
  exit 2
fi
if [ -e "$CLAIM" ] || [ -L "$CLAIM" ]; then
  echo "score controller already claimed" >&2
  exit 2
fi

"$PYTHON" - \
  "$CLAIM" "$source_commit" "$ALLOCATION_JOB_ID" "$ARTIFACT_ROOT" \
  "$BOARD_ROOT" "$SCORER" <<'PY'
import json
import os
import sys

path, source_commit, allocation_job_id, artifact_root, board_root, scorer = sys.argv[1:]
payload = {
    "schema": "shohin-dense-public-host-score-controller-v1",
    "status": "waiting_for_generation",
    "source_commit": source_commit,
    "allocation_job_id": int(allocation_job_id),
    "artifact_root": artifact_root,
    "board_root": board_root,
    "scorer": scorer,
    "controller_pid": os.getppid(),
    "duplicate_scoring": False,
    "requeue": False,
}
data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(fd, "wb") as handle:
    handle.write(data)
    handle.flush()
    os.fsync(handle.fileno())
PY

while ! squeue -h -j "$ALLOCATION_JOB_ID" -t RUNNING | grep -q .; do
  state=$(sacct -X -n -j "$ALLOCATION_JOB_ID" --format=State -P | head -n 1)
  case "$state" in
    COMPLETED*|FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*)
      echo "allocation ended before host scoring began: $state" >&2
      exit 2
      ;;
  esac
  sleep "$POLL_SECONDS"
done

exec env BOARD_ROOT="$BOARD_ROOT" "$SCORER"
