#!/bin/bash
# Join healthy H100 groups from one or more live reservations into one
# synchronized progressive-coupling run. Each group is JOB@NODE@GPUS.

set -euo pipefail

ALLOCATION_GROUPS=${ALLOCATION_GROUPS:?set comma-separated JOB@NODE@GPUS groups}
CODE_ROOT=${CODE_ROOT:?set the immutable shared source root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set the exact private source commit}
RELEASE_ROOT=${RELEASE_ROOT:?set the immutable ETTR release root}
RELEASE_SHA256=${RELEASE_SHA256:?set the release.json SHA-256}
DATA_ROOT=${DATA_ROOT:?set the immutable ETTR shard root}
TOKENIZER=${TOKENIZER:?set the immutable tokenizer path}
PROTECTED_CHECKPOINT=${PROTECTED_CHECKPOINT:?set the protected 300k checkpoint}
CHECKPOINT=${CHECKPOINT:?set the architecture checkpoint}
CHECKPOINT_SHA256=${CHECKPOINT_SHA256:?set the architecture checkpoint SHA-256}
RUN_CONTRACT=${RUN_CONTRACT:?set the architecture run contract}
RUN_CONTRACT_SHA256=${RUN_CONTRACT_SHA256:?set the run contract SHA-256}
INITIAL_COMPILER=${INITIAL_COMPILER:?set the compiler warm start}
INITIAL_COMPILER_SHA256=${INITIAL_COMPILER_SHA256:?set its SHA-256}
INITIAL_REACTOR=${INITIAL_REACTOR:?set the reactor warm start}
INITIAL_REACTOR_SHA256=${INITIAL_REACTOR_SHA256:?set its SHA-256}
INITIAL_READER=${INITIAL_READER:?set the reader warm start}
INITIAL_READER_SHA256=${INITIAL_READER_SHA256:?set its SHA-256}
OUTDIR=${OUTDIR:?set a fresh isolated output directory}
ARCHITECTURE_SEED=${ARCHITECTURE_SEED:?set the architecture seed}
DATA_SEED=${DATA_SEED:?set the data seed}
COUPLING_SEED=${COUPLING_SEED:?set the coupling seed}
UPDATES=${UPDATES:-100}
START_POSITION=${START_POSITION:-20000}
WARMUP_UPDATES=${WARMUP_UPDATES:-10}
RAMP_UPDATES=${RAMP_UPDATES:-70}
COUNTERFACTUAL_DELTA_WEIGHT=${COUNTERFACTUAL_DELTA_WEIGHT:-2}
EXACT_ANCHOR_STEPS=${EXACT_ANCHOR_STEPS:-4}
CREDIT_HORIZON=${CREDIT_HORIZON:-4}
COMPILER_LEARNING_RATE=${COMPILER_LEARNING_RATE:-3e-5}
REACTOR_LEARNING_RATE=${REACTOR_LEARNING_RATE:-5e-5}
READER_LEARNING_RATE=${READER_LEARNING_RATE:-3e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0}
GRADIENT_CLIP=${GRADIENT_CLIP:-1}
EVAL_BATCHES=${EVAL_BATCHES:-16}
LOG_EVERY=${LOG_EVERY:-10}
CPUS_PER_GPU=${CPUS_PER_GPU:-2}
PYTHON_ROOT=${PYTHON_ROOT:-/lustre/fs1/home/sa305415/shohin/miniforge3}

integer_contract="$ARCHITECTURE_SEED:$DATA_SEED:$COUPLING_SEED:$UPDATES"
integer_contract+=":$START_POSITION:$WARMUP_UPDATES:$RAMP_UPDATES"
integer_contract+=":$EXACT_ANCHOR_STEPS:$CREDIT_HORIZON:$EVAL_BATCHES"
integer_contract+=":$LOG_EVERY:$CPUS_PER_GPU"
case "$integer_contract" in
  *[!0-9:]* | *::* | :* | *:)
    echo "integer coupling launch settings differ" >&2
    exit 2
    ;;
esac
if (( UPDATES < 1 \
  || WARMUP_UPDATES + RAMP_UPDATES > UPDATES \
  || CPUS_PER_GPU < 1 )); then
  echo "coupling schedule or CPU allocation differs" >&2
  exit 2
fi
if [[ ! "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "source commit differs" >&2
  exit 2
fi
for value in \
  "$RELEASE_SHA256" \
  "$CHECKPOINT_SHA256" \
  "$RUN_CONTRACT_SHA256" \
  "$INITIAL_COMPILER_SHA256" \
  "$INITIAL_REACTOR_SHA256" \
  "$INITIAL_READER_SHA256"; do
  if [[ ! "$value" =~ ^[0-9a-f]{64}$ ]]; then
    echo "coupling input hash differs" >&2
    exit 2
  fi
done
for path in \
  "$CODE_ROOT" \
  "$RELEASE_ROOT" \
  "$DATA_ROOT" \
  "$TOKENIZER" \
  "$PROTECTED_CHECKPOINT" \
  "$CHECKPOINT" \
  "$RUN_CONTRACT" \
  "$INITIAL_COMPILER" \
  "$INITIAL_REACTOR" \
  "$INITIAL_READER" \
  "$OUTDIR" \
  "$PYTHON_ROOT"; do
  if [[ "$path" != /* ]]; then
    echo "all coupling launch paths must be absolute: $path" >&2
    exit 2
  fi
done
launcher_root="${OUTDIR}.launcher"
if [[ -e "$OUTDIR" || -L "$OUTDIR" \
  || -e "$launcher_root" || -L "$launcher_root" ]]; then
  echo "refusing existing coupling output or launcher root" >&2
  exit 2
fi

IFS=',' read -r -a group_specs <<< "$ALLOCATION_GROUPS"
if (( ${#group_specs[@]} < 1 || ${#group_specs[@]} > 20 )); then
  echo "coupling allocation group count differs" >&2
  exit 2
fi
jobs=()
nodes=()
gpu_counts=()
rank_offsets=()
declare -A seen_groups=()
world_size=0
for spec in "${group_specs[@]}"; do
  IFS='@' read -r job node gpus extra <<< "$spec"
  if [[ -n "${extra:-}" \
    || ! "$job" =~ ^[0-9]+$ \
    || ! "$node" =~ ^[A-Za-z0-9._-]+$ \
    || ! "$gpus" =~ ^[0-9]+$ \
    || "$gpus" == 0 \
    || -n "${seen_groups[$job@$node]:-}" ]]; then
    echo "coupling allocation group differs: $spec" >&2
    exit 2
  fi
  seen_groups[$job@$node]=1
  state=$(squeue -h -j "$job" -o "%T")
  allocation_nodes=$(squeue -h -j "$job" -o "%N")
  if [[ "$state" != RUNNING ]] \
    || ! scontrol show hostnames "$allocation_nodes" | grep -Fxq "$node"; then
    echo "coupling allocation is not live on requested node: $spec" >&2
    exit 2
  fi
  jobs+=("$job")
  nodes+=("$node")
  gpu_counts+=("$gpus")
  rank_offsets+=("$world_size")
  world_size=$((world_size + gpus))
done
if (( world_size < 2 || world_size > 20 \
  || START_POSITION % world_size != 0 )); then
  echo "coupling world size or start cursor differs" >&2
  exit 2
fi

cd "$CODE_ROOT"
if [[ ! -r SOURCE_COMMIT \
  || "$(tr -d '\r\n' < SOURCE_COMMIT)" != "$SOURCE_COMMIT" \
  || ! -r SHA256SUMS \
  || -n "$(sha256sum -c SHA256SUMS 2>&1 | grep -v ': OK$')" ]]; then
  echo "coupling source archive differs" >&2
  exit 2
fi
if [[ ! -s "$RELEASE_ROOT/release.json" \
  || "$(sha256sum "$RELEASE_ROOT/release.json" | awk '{print $1}')" \
    != "$RELEASE_SHA256" ]]; then
  echo "ETTR release identity differs" >&2
  exit 2
fi

mkdir -m 700 "$launcher_root"
master_addr=${nodes[0]}
master_port=$((20000 + jobs[0] % 20000))
export MASTER_ADDR="$master_addr"
export MASTER_PORT="$master_port"
export WORLD_SIZE="$world_size"
export CODE_ROOT SOURCE_COMMIT RELEASE_ROOT RELEASE_SHA256 DATA_ROOT TOKENIZER
export PROTECTED_CHECKPOINT CHECKPOINT CHECKPOINT_SHA256
export RUN_CONTRACT RUN_CONTRACT_SHA256
export INITIAL_COMPILER INITIAL_COMPILER_SHA256
export INITIAL_REACTOR INITIAL_REACTOR_SHA256
export INITIAL_READER INITIAL_READER_SHA256
export OUTDIR ARCHITECTURE_SEED DATA_SEED COUPLING_SEED
export UPDATES START_POSITION WARMUP_UPDATES RAMP_UPDATES
export COUNTERFACTUAL_DELTA_WEIGHT EXACT_ANCHOR_STEPS CREDIT_HORIZON
export COMPILER_LEARNING_RATE REACTOR_LEARNING_RATE READER_LEARNING_RATE
export WEIGHT_DECAY GRADIENT_CLIP EVAL_BATCHES LOG_EVERY PYTHON_ROOT
export OMP_NUM_THREADS="$CPUS_PER_GPU"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
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
  'federated_coupling world=%s groups=%s commit=%s updates=%s\n' \
  "$world_size" "$ALLOCATION_GROUPS" "$SOURCE_COMMIT" "$UPDATES"

for index in "${!jobs[@]}"; do
  job=${jobs[$index]}
  node=${nodes[$index]}
  gpus=${gpu_counts[$index]}
  rank_offset=${rank_offsets[$index]}
  cpus=$((gpus * CPUS_PER_GPU))
  (
    export GROUP_GPUS="$gpus"
    export RANK_OFFSET="$rank_offset"
    export LAUNCHER_ROOT="$launcher_root"
    srun \
      --jobid="$job" \
      --overlap \
      --nodes=1 \
      --nodelist="$node" \
      --ntasks=1 \
      --cpus-per-task="$cpus" \
      --gpus-per-node="$gpus" \
      --kill-on-bad-exit=1 \
      bash -lc '
        set -euo pipefail
        test "$("$PYTHON_ROOT/bin/python" -c "import torch; print(torch.cuda.device_count())")" \
          -eq "$GROUP_GPUS"
        test -n "$(ls -A /sys/class/infiniband 2>/dev/null)"
        local_pids=()
        stop_local() {
          local pid
          for pid in "${local_pids[@]:-}"; do
            kill "$pid" 2>/dev/null || true
          done
          wait 2>/dev/null || true
        }
        trap stop_local INT TERM
        for ((local_rank = 0; local_rank < GROUP_GPUS; local_rank++)); do
          rank=$((RANK_OFFSET + local_rank))
          (
            export RANK="$rank"
            export LOCAL_RANK="$local_rank"
            "$PYTHON_ROOT/bin/python" \
              "$CODE_ROOT/train/train_ettr_progressive_coupling.py" \
              --release-root "$RELEASE_ROOT" \
              --release-sha256 "$RELEASE_SHA256" \
              --data-root "$DATA_ROOT" \
              --tokenizer "$TOKENIZER" \
              --protected-checkpoint "$PROTECTED_CHECKPOINT" \
              --checkpoint "$CHECKPOINT" \
              --checkpoint-sha256 "$CHECKPOINT_SHA256" \
              --run-contract "$RUN_CONTRACT" \
              --run-contract-sha256 "$RUN_CONTRACT_SHA256" \
              --initial-compiler "$INITIAL_COMPILER" \
              --initial-compiler-sha256 "$INITIAL_COMPILER_SHA256" \
              --initial-reactor "$INITIAL_REACTOR" \
              --initial-reactor-sha256 "$INITIAL_REACTOR_SHA256" \
              --initial-reader "$INITIAL_READER" \
              --initial-reader-sha256 "$INITIAL_READER_SHA256" \
              --compiler-learning-rate "$COMPILER_LEARNING_RATE" \
              --reactor-learning-rate "$REACTOR_LEARNING_RATE" \
              --reader-learning-rate "$READER_LEARNING_RATE" \
              --output "$OUTDIR" \
              --source-commit "$SOURCE_COMMIT" \
              --architecture-seed "$ARCHITECTURE_SEED" \
              --data-seed "$DATA_SEED" \
              --coupling-seed "$COUPLING_SEED" \
              --updates "$UPDATES" \
              --start-position "$START_POSITION" \
              --warmup-updates "$WARMUP_UPDATES" \
              --ramp-updates "$RAMP_UPDATES" \
              --counterfactual-delta-weight "$COUNTERFACTUAL_DELTA_WEIGHT" \
              --exact-anchor-steps "$EXACT_ANCHOR_STEPS" \
              --credit-horizon "$CREDIT_HORIZON" \
              --weight-decay "$WEIGHT_DECAY" \
              --gradient-clip "$GRADIENT_CLIP" \
              --eval-batches "$EVAL_BATCHES" \
              --log-every "$LOG_EVERY"
          ) >"$LAUNCHER_ROOT/rank-$(printf "%03d" "$rank").log" 2>&1 &
          local_pids+=("$!")
        done
        result=0
        remaining=${#local_pids[@]}
        while (( remaining > 0 )); do
          if ! wait -n; then
            result=1
            break
          fi
          remaining=$((remaining - 1))
        done
        if (( result != 0 )); then
          stop_local
          exit 1
        fi
        wait
        trap - INT TERM
      '
  ) >"$launcher_root/group-$(printf '%03d' "$index").log" 2>&1 &
  pids+=("$!")
done

result=0
remaining=${#pids[@]}
while (( remaining > 0 )); do
  if ! wait -n; then
    result=1
    break
  fi
  remaining=$((remaining - 1))
done
if (( result != 0 )); then
  terminate_steps
  echo "federated coupling rank failed; reservations remain alive" >&2
  exit 1
fi
wait
trap - INT TERM

test -s "$OUTDIR/report.json"
test -s "$OUTDIR/compiler-final.safetensors"
test -s "$OUTDIR/reactor-final.safetensors"
test -s "$OUTDIR/reader-final.safetensors"
printf 'federated coupling completed: %s\n' "$OUTDIR"
