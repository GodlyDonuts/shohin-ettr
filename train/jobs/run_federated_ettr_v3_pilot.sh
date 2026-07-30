#!/bin/bash
# Join independent one-H100 reservations into one bounded ETTR-v3 DDP rung.
# Every reservation is validated independently. A rank failure kills only the
# launched job steps and leaves the reservation jobs alive for diagnosis.

set -euo pipefail

ALLOCATION_JOB_IDS=${ALLOCATION_JOB_IDS:?set comma-separated running job IDs}
CODE_ROOT=${CODE_ROOT:?set the immutable shared source root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set the exact private source commit}
RELEASE_ROOT=${RELEASE_ROOT:?set the immutable ETTR release root}
RELEASE_SHA256=${RELEASE_SHA256:?set the release.json SHA-256}
DATA_ROOT=${DATA_ROOT:?set the immutable ETTR shard root}
TOKENIZER=${TOKENIZER:?set the immutable tokenizer path}
PROTECTED_CHECKPOINT=${PROTECTED_CHECKPOINT:?set the protected 300k checkpoint}
OUTDIR=${OUTDIR:?set a fresh isolated pilot output directory}
START_UPDATE=${START_UPDATE:-0}
TARGET_UPDATE=${TARGET_UPDATE:-100}
ACCUMULATION=${ACCUMULATION:-1}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-100}
LOG_EVERY=${LOG_EVERY:-10}
MAX_EVAL_BATCHES=${MAX_EVAL_BATCHES:-64}
ARCHITECTURE_SEED=${ARCHITECTURE_SEED:-2026072801}
DATA_SEED=${DATA_SEED:-2026072802}
TOTAL_UPDATES=${TOTAL_UPDATES:-300000}
WARMUP_UPDATES=${WARMUP_UPDATES:-2000}
FREEZE_BASE=${FREEZE_BASE:-1}
COMPILE_MODE=${COMPILE_MODE:-default}
LAUNCH_STAGGER_SECONDS=${LAUNCH_STAGGER_SECONDS:-1}
PYTHON_ROOT=${PYTHON_ROOT:-/lustre/fs1/home/sa305415/shohin/miniforge3}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}
RESUME_SHA256=${RESUME_SHA256:-}

integer_contract="$START_UPDATE:$TARGET_UPDATE:$ACCUMULATION"
integer_contract+=":$CHECKPOINT_EVERY:$LOG_EVERY:$MAX_EVAL_BATCHES"
integer_contract+=":$ARCHITECTURE_SEED:$DATA_SEED:$TOTAL_UPDATES"
integer_contract+=":$WARMUP_UPDATES:$FREEZE_BASE"
case "$integer_contract" in
  *[!0-9:]* | *::* | :* | *:)
    echo "integer federated launch settings differ" >&2
    exit 2
    ;;
esac
if (( TARGET_UPDATE <= START_UPDATE || TARGET_UPDATE > TOTAL_UPDATES )); then
  echo "federated ETTR update range differs" >&2
  exit 2
fi
UPDATES=$((TARGET_UPDATE - START_UPDATE))
if [[ ! "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
  || [[ ! "$RELEASE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "federated source or release identity differs" >&2
  exit 2
fi
if [[ "$FREEZE_BASE" != 0 && "$FREEZE_BASE" != 1 ]]; then
  echo "federated freeze-base flag differs" >&2
  exit 2
fi
if [[ ! "$LAUNCH_STAGGER_SECONDS" =~ ^[0-9]+$ ]] \
  || (( LAUNCH_STAGGER_SECONDS > 10 )); then
  echo "federated launch stagger differs" >&2
  exit 2
fi
if [[ "$COMPILE_MODE" != default \
  && "$COMPILE_MODE" != reduce-overhead \
  && "$COMPILE_MODE" != max-autotune \
  && "$COMPILE_MODE" != max-autotune-no-cudagraphs ]]; then
  echo "federated compile mode differs" >&2
  exit 2
fi
for path in \
  "$CODE_ROOT" \
  "$RELEASE_ROOT" \
  "$DATA_ROOT" \
  "$TOKENIZER" \
  "$PROTECTED_CHECKPOINT" \
  "$OUTDIR" \
  "$PYTHON_ROOT"; do
  if [[ "$path" != /* ]]; then
    echo "all federated pilot paths must be absolute: $path" >&2
    exit 2
  fi
done
if [[ -e "$OUTDIR" || -L "$OUTDIR" ]]; then
  echo "refusing existing federated ETTR pilot output" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_ROOT/bin/torchrun" ]]; then
  echo "torchrun is unavailable" >&2
  exit 2
fi
if [[ "$START_UPDATE" == 0 ]]; then
  if [[ -n "$RESUME_CHECKPOINT" || -n "$RESUME_SHA256" ]]; then
    echo "initial federated rung may not provide resume state" >&2
    exit 2
  fi
else
  if [[ "$RESUME_CHECKPOINT" != /* \
    || ! "$RESUME_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "federated continuation requires exact resume state" >&2
    exit 2
  fi
fi

IFS=',' read -r -a job_ids <<< "$ALLOCATION_JOB_IDS"
world_size=${#job_ids[@]}
if (( world_size < 2 || world_size > 20 )); then
  echo "federated ETTR world size differs" >&2
  exit 2
fi
declare -A seen_jobs=()
nodes=()
for job in "${job_ids[@]}"; do
  if [[ ! "$job" =~ ^[0-9]+$ || -n "${seen_jobs[$job]:-}" ]]; then
    echo "federated ETTR job identity differs" >&2
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
    || "$node" == *"["* ]]; then
    echo "federated ETTR reservation geometry differs: $job" >&2
    exit 2
  fi
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
if [[ ! -s "$RELEASE_ROOT/release.json" \
  || "$(sha256sum "$RELEASE_ROOT/release.json" | awk '{print $1}')" \
    != "$RELEASE_SHA256" ]]; then
  echo "ETTR release identity differs" >&2
  exit 2
fi

master_addr=${nodes[0]}
master_port=$((20000 + job_ids[0] % 20000))
rdzv_id="shohin-federated-${job_ids[0]}-ettr-${TARGET_UPDATE}"
mkdir -m 700 "$OUTDIR"
mkdir -m 700 "$OUTDIR/launcher"

export CODE_ROOT SOURCE_COMMIT RELEASE_ROOT RELEASE_SHA256 DATA_ROOT TOKENIZER
export PROTECTED_CHECKPOINT OUTDIR START_UPDATE TARGET_UPDATE UPDATES
export ACCUMULATION CHECKPOINT_EVERY LOG_EVERY MAX_EVAL_BATCHES
export ARCHITECTURE_SEED DATA_SEED TOTAL_UPDATES WARMUP_UPDATES FREEZE_BASE
export COMPILE_MODE PYTHON_ROOT RESUME_CHECKPOINT RESUME_SHA256
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
  'federated_ettr_pilot world=%s jobs=%s nodes=%s updates=%s..%s commit=%s\n' \
  "$world_size" "$ALLOCATION_JOB_IDS" "$(IFS=,; echo "${nodes[*]}")" \
  "$START_UPDATE" "$TARGET_UPDATE" "$SOURCE_COMMIT"

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
        freeze_args=()
        resume_args=()
        if [[ "$FREEZE_BASE" == 1 ]]; then
          freeze_args+=(--freeze-base)
        fi
        if [[ "$START_UPDATE" != 0 ]]; then
          resume_args+=(
            --resume-checkpoint "$RESUME_CHECKPOINT"
            --resume-sha256 "$RESUME_SHA256"
          )
        fi
        "$PYTHON_ROOT/bin/torchrun" \
          --nnodes="$world_size" \
          --nproc_per_node=1 \
          --node_rank="$FEDERATED_RANK" \
          --rdzv_backend=c10d \
          --rdzv_endpoint="$master_addr:$master_port" \
          --rdzv_id="$rdzv_id" \
          "$CODE_ROOT/train/train_ettr_v3.py" \
          --release-root "$RELEASE_ROOT" \
          --release-sha256 "$RELEASE_SHA256" \
          --data-root "$DATA_ROOT" \
          --tokenizer "$TOKENIZER" \
          --protected-checkpoint "$PROTECTED_CHECKPOINT" \
          --output "$OUTDIR/train" \
          --source-commit "$SOURCE_COMMIT" \
          --updates "$UPDATES" \
          --accumulation "$ACCUMULATION" \
          --checkpoint-every "$CHECKPOINT_EVERY" \
          --log-every "$LOG_EVERY" \
          --architecture-seed "$ARCHITECTURE_SEED" \
          --data-seed "$DATA_SEED" \
          --total-updates "$TOTAL_UPDATES" \
          --warmup-updates "$WARMUP_UPDATES" \
          --compile-mode "$COMPILE_MODE" \
          "${freeze_args[@]}" \
          "${resume_args[@]}"
      '
  ) > "$OUTDIR/launcher/rank-$(printf '%03d' "$rank").log" 2>&1 &
  pids+=("$!")
  if (( rank + 1 < world_size && LAUNCH_STAGGER_SECONDS > 0 )); then
    sleep "$LAUNCH_STAGGER_SECONDS"
  fi
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
  echo "federated ETTR rank failed; reservations remain alive" >&2
  exit 1
fi
wait
trap - INT TERM

checkpoint=$(printf \
  '%s/train/checkpoint-update-%07d.pt' \
  "$OUTDIR" \
  "$TARGET_UPDATE")
sidecar=${checkpoint%.pt}.json
contract="$OUTDIR/train/run-contract.json"
test -s "$checkpoint"
test -s "$sidecar"
test -s "$contract"
checkpoint_sha=$(
  "$PYTHON_ROOT/bin/python" -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["checkpoint_sha256"])' \
    "$sidecar"
)
contract_sha=$(sha256sum "$contract" | awk '{print $1}')
if [[ ! "$checkpoint_sha" =~ ^[0-9a-f]{64}$ \
  || ! "$contract_sha" =~ ^[0-9a-f]{64}$ ]]; then
  echo "federated checkpoint or run-contract identity differs" >&2
  exit 2
fi

srun \
  --jobid="${job_ids[0]}" \
  --overlap \
  --nodes=1 \
  --nodelist="${nodes[0]}" \
  --ntasks=1 \
  --cpus-per-task=4 \
  --gpus-per-node=1 \
  --kill-on-bad-exit=1 \
  "$PYTHON_ROOT/bin/python" \
    "$CODE_ROOT/train/eval_ettr_v3.py" \
    --release-root "$RELEASE_ROOT" \
    --release-sha256 "$RELEASE_SHA256" \
    --data-root "$DATA_ROOT" \
    --tokenizer "$TOKENIZER" \
    --protected-checkpoint "$PROTECTED_CHECKPOINT" \
    --output "$OUTDIR/development-evaluation.json" \
    --source-commit "$SOURCE_COMMIT" \
    --architecture-seed "$ARCHITECTURE_SEED" \
    --data-seed "$DATA_SEED" \
    --max-batches "$MAX_EVAL_BATCHES" \
    --checkpoint "$checkpoint" \
    --checkpoint-sha256 "$checkpoint_sha" \
    --run-contract "$contract" \
    --run-contract-sha256 "$contract_sha"

sha256sum \
  "$checkpoint" \
  "$sidecar" \
  "$contract" \
  "$OUTDIR/development-evaluation.json" \
  "$OUTDIR/launcher"/*.log \
  > "$OUTDIR/SHA256SUMS"
chmod 400 \
  "$checkpoint" \
  "$sidecar" \
  "$contract" \
  "$OUTDIR/development-evaluation.json" \
  "$OUTDIR/SHA256SUMS" \
  "$OUTDIR/launcher"/*.log
chmod 500 "$OUTDIR/launcher"
printf \
  'federated_ettr_pilot_complete output=%s target_update=%s world=%s\n' \
  "$OUTDIR" "$TARGET_UPDATE" "$world_size"
