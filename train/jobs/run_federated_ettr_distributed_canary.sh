#!/bin/bash
# Join independent one-H100 reservations into one fixed ETTR DDP canary world.
# The selected jobs remain reservations; only the bounded job steps are killed
# if any rank fails.

set -euo pipefail

ALLOCATION_JOB_IDS=${ALLOCATION_JOB_IDS:?set comma-separated running job IDs}
CODE_ROOT=${CODE_ROOT:?set the immutable shared source root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set the exact private source commit}
PROTECTED_CHECKPOINT=${PROTECTED_CHECKPOINT:?set the protected 300k checkpoint}
PROTECTED_CHECKPOINT_SHA256=${PROTECTED_CHECKPOINT_SHA256:?set its SHA-256}
OUTDIR=${OUTDIR:?set a fresh isolated canary output directory}
COMPILE_MODE=${COMPILE_MODE:-default}
PYTHON_ROOT=${PYTHON_ROOT:-/lustre/fs1/home/sa305415/shohin/miniforge3}

if [[ ! "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
  || [[ ! "$PROTECTED_CHECKPOINT_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "federated canary source or checkpoint identity differs" >&2
  exit 2
fi
if [[ "$COMPILE_MODE" != default \
  && "$COMPILE_MODE" != reduce-overhead \
  && "$COMPILE_MODE" != max-autotune ]]; then
  echo "federated canary compile mode differs" >&2
  exit 2
fi
for path in "$CODE_ROOT" "$PROTECTED_CHECKPOINT" "$OUTDIR" "$PYTHON_ROOT"; do
  if [[ "$path" != /* ]]; then
    echo "all federated canary paths must be absolute: $path" >&2
    exit 2
  fi
done
if [[ -e "$OUTDIR" || -L "$OUTDIR" \
  || -e "$OUTDIR.launcher" || -L "$OUTDIR.launcher" ]]; then
  echo "refusing existing federated canary output" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_ROOT/bin/torchrun" ]]; then
  echo "torchrun is unavailable" >&2
  exit 2
fi

IFS=',' read -r -a job_ids <<< "$ALLOCATION_JOB_IDS"
world_size=${#job_ids[@]}
if (( world_size < 2 || world_size > 20 )); then
  echo "federated canary world size differs" >&2
  exit 2
fi
declare -A seen_jobs=()
declare -A seen_nodes=()
nodes=()
for job in "${job_ids[@]}"; do
  if [[ ! "$job" =~ ^[0-9]+$ || -n "${seen_jobs[$job]:-}" ]]; then
    echo "federated canary job identity differs" >&2
    exit 2
  fi
  seen_jobs[$job]=1
  state=$(squeue -h -j "$job" -o "%T")
  name=$(squeue -h -j "$job" -o "%j")
  node_count=$(squeue -h -j "$job" -o "%D")
  node=$(squeue -h -j "$job" -o "%N")
  if [[ "$state" != RUNNING \
    || "$name" != shohin-1h100-* \
    || "$node_count" != 1 \
    || -z "$node" \
    || "$node" == *"["* \
    || -n "${seen_nodes[$node]:-}" ]]; then
    echo "federated canary reservation geometry differs: $job" >&2
    exit 2
  fi
  seen_nodes[$node]=1
  nodes+=("$node")
done

cd "$CODE_ROOT"
if [[ ! -r SOURCE_COMMIT \
  || "$(tr -d '\r\n' < SOURCE_COMMIT)" != "$SOURCE_COMMIT" ]]; then
  echo "source archive commit differs" >&2
  exit 2
fi
if [[ ! -r SHA256SUMS \
  || -n "$(sha256sum -c SHA256SUMS 2>&1 | grep -v ': OK$')" ]]; then
  echo "source archive digest differs" >&2
  exit 2
fi
if [[ ! -s "$PROTECTED_CHECKPOINT" \
  || "$(sha256sum "$PROTECTED_CHECKPOINT" | awk '{print $1}')" \
    != "$PROTECTED_CHECKPOINT_SHA256" ]]; then
  echo "protected checkpoint identity differs" >&2
  exit 2
fi

master_addr=${nodes[0]}
master_port=$((30000 + job_ids[0] % 20000))
rdzv_id="shohin-federated-${job_ids[0]}-ettr-canary"
logdir="$OUTDIR.launcher"
mkdir -m 700 "$logdir"
export CODE_ROOT SOURCE_COMMIT PROTECTED_CHECKPOINT
export PROTECTED_CHECKPOINT_SHA256 OUTDIR COMPILE_MODE PYTHON_ROOT
export master_addr master_port rdzv_id world_size
export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=0

pids=()
terminate_steps() {
  local pid
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap terminate_steps INT TERM

printf \
  'federated_ettr_canary world=%s jobs=%s nodes=%s commit=%s\n' \
  "$world_size" "$ALLOCATION_JOB_IDS" "$(IFS=,; echo "${nodes[*]}")" \
  "$SOURCE_COMMIT"

for rank in "${!job_ids[@]}"; do
  job=${job_ids[$rank]}
  node=${nodes[$rank]}
  (
    export FEDERATED_RANK="$rank"
    srun \
      --jobid="$job" \
      --overlap \
      --nodes=1 \
      --nodelist="$node" \
      --ntasks=1 \
      --cpus-per-task=4 \
      --gpus-per-node=1 \
      --kill-on-bad-exit=1 \
      bash -lc '
        set -euo pipefail
        test "$(nvidia-smi -L | wc -l)" -eq 1
        test -n "$(ls -A /sys/class/infiniband 2>/dev/null)"
        "$PYTHON_ROOT/bin/torchrun" \
          --nnodes="$world_size" \
          --nproc_per_node=1 \
          --node_rank="$FEDERATED_RANK" \
          --rdzv_backend=c10d \
          --rdzv_endpoint="$master_addr:$master_port" \
          --rdzv_id="$rdzv_id" \
          "$CODE_ROOT/train/canary_ettr_distributed_h100.py" \
          --output "$OUTDIR" \
          --checkpoint "$PROTECTED_CHECKPOINT" \
          --checkpoint-sha256 "$PROTECTED_CHECKPOINT_SHA256" \
          --expected-step 300000 \
          --source-commit "$SOURCE_COMMIT" \
          --expected-world-size "$world_size" \
          --compile-mode "$COMPILE_MODE"
      '
  ) > "$logdir/rank-$(printf '%03d' "$rank").log" 2>&1 &
  pids+=("$!")
done

result=0
remaining=$world_size
while (( remaining > 0 )); do
  if ! wait -n; then
    result=1
    break
  fi
  remaining=$((remaining - 1))
done
if (( result != 0 )); then
  terminate_steps
  echo "federated canary rank failed; reservations remain alive" >&2
  exit 1
fi
wait
trap - INT TERM

test -s "$OUTDIR/report.json"
if [[ ! -s "$OUTDIR/SHA256SUMS" ]]; then
  chmod 700 "$OUTDIR"
  test ! -e "$OUTDIR/SHA256SUMS"
  (
    cd "$OUTDIR"
    sha256sum report.json > SHA256SUMS
    chmod 400 report.json SHA256SUMS
  )
  chmod 500 "$OUTDIR"
fi
(
  cd "$OUTDIR"
  sha256sum -c SHA256SUMS
)
chmod 400 "$logdir"/*.log
chmod 500 "$logdir"
printf 'federated_ettr_canary_complete output=%s\n' "$OUTDIR"
