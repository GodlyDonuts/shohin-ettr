#!/bin/bash
# Publish one hash-bound ETTR rank command to each dispatchable H100
# reservation. The ranks run directly as their Slurm batch payloads, avoiding
# one login-node srun client per GPU.

set -euo pipefail
umask 077

ALLOCATION_GROUPS=${ALLOCATION_GROUPS:?set comma-separated JOB@NODE groups}
CODE_ROOT=${CODE_ROOT:?set the immutable shared source root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set the exact private source commit}
RELEASE_ROOT=${RELEASE_ROOT:?set the immutable ETTR release root}
RELEASE_SHA256=${RELEASE_SHA256:?set the release.json SHA-256}
DATA_ROOT=${DATA_ROOT:?set the immutable ETTR shard root}
TOKENIZER=${TOKENIZER:?set the immutable tokenizer path}
PROTECTED_CHECKPOINT=${PROTECTED_CHECKPOINT:?set the protected checkpoint}
CHECKPOINT=${CHECKPOINT:?set the architecture checkpoint}
CHECKPOINT_SHA256=${CHECKPOINT_SHA256:?set its SHA-256}
RUN_CONTRACT=${RUN_CONTRACT:?set the architecture run contract}
RUN_CONTRACT_SHA256=${RUN_CONTRACT_SHA256:?set its SHA-256}
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
PYTHON_ROOT=${PYTHON_ROOT:-/lustre/fs1/home/sa305415/shohin/miniforge3}
CONTROL_ROOT=/lustre/fs1/home/sa305415/shohin/control/ettr-h100-dispatch-v1

integer_contract="$ARCHITECTURE_SEED:$DATA_SEED:$COUPLING_SEED:$UPDATES"
integer_contract+=":$START_POSITION:$WARMUP_UPDATES:$RAMP_UPDATES"
integer_contract+=":$EXACT_ANCHOR_STEPS:$CREDIT_HORIZON:$EVAL_BATCHES"
integer_contract+=":$LOG_EVERY"
case "$integer_contract" in
  *[!0-9:]* | *::* | :* | *:)
    echo "dispatch integer settings differ" >&2
    exit 2
    ;;
esac
if (( UPDATES < 1 || WARMUP_UPDATES + RAMP_UPDATES > UPDATES )); then
  echo "dispatch coupling schedule differs" >&2
  exit 2
fi
if [[ ! "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "dispatch source commit differs" >&2
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
    echo "dispatch input hash differs" >&2
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
    echo "dispatch paths must be absolute: $path" >&2
    exit 2
  fi
done

IFS=',' read -r -a group_specs <<< "$ALLOCATION_GROUPS"
world_size=${#group_specs[@]}
if (( world_size < 2 || world_size > 20 \
  || START_POSITION % world_size != 0 )); then
  echo "dispatch world size or data cursor differs" >&2
  exit 2
fi
jobs=()
nodes=()
declare -A seen=()
for spec in "${group_specs[@]}"; do
  IFS='@' read -r job node extra <<< "$spec"
  if [[ -n "${extra:-}" \
    || ! "$job" =~ ^[0-9]+$ \
    || ! "$node" =~ ^[A-Za-z0-9._-]+$ \
    || -n "${seen[$job]:-}" ]]; then
    echo "dispatch allocation group differs: $spec" >&2
    exit 2
  fi
  seen[$job]=1
  if [[ "$(squeue -h -j "$job" -o '%T')" != RUNNING \
    || "$(squeue -h -j "$job" -o '%N')" != "$node" \
    || ! -f "$CONTROL_ROOT/ready/$job" \
    || -e "$CONTROL_ROOT/commands/$job.sh" \
    || -L "$CONTROL_ROOT/commands/$job.sh" \
    || -e "$CONTROL_ROOT/commands/$job.sha256" \
    || -L "$CONTROL_ROOT/commands/$job.sha256" \
    || -e "$CONTROL_ROOT/status/$job" \
    || -L "$CONTROL_ROOT/status/$job" ]]; then
    echo "dispatch reservation is not fresh and ready: $spec" >&2
    exit 2
  fi
  jobs+=("$job")
  nodes+=("$node")
done

cd "$CODE_ROOT"
if [[ ! -r SOURCE_COMMIT \
  || "$(tr -d '\r\n' <SOURCE_COMMIT)" != "$SOURCE_COMMIT" \
  || ! -r SHA256SUMS \
  || -n "$(sha256sum -c SHA256SUMS 2>&1 | grep -v ': OK$')" \
  || ! -s "$RELEASE_ROOT/release.json" \
  || "$(sha256sum "$RELEASE_ROOT/release.json" | awk '{print $1}')" \
    != "$RELEASE_SHA256" ]]; then
  echo "dispatch source or release identity differs" >&2
  exit 2
fi

launcher_root="${OUTDIR}.launcher"
if [[ -e "$OUTDIR" || -L "$OUTDIR" \
  || -e "$launcher_root" || -L "$launcher_root" ]]; then
  echo "dispatch output or launcher root already exists" >&2
  exit 2
fi
mkdir -m 700 "$launcher_root"
master_addr=${nodes[0]}
master_port=$((20000 + jobs[0] % 20000))

emit_export() {
  local name=$1
  local value=$2
  printf 'export %s=' "$name"
  printf '%q' "$value"
  printf '\n'
}

for rank in "${!jobs[@]}"; do
  job=${jobs[$rank]}
  staged="$launcher_root/rank-$(printf '%03d' "$rank").command"
  rank_log="$launcher_root/rank-$(printf '%03d' "$rank").log"
  {
    printf '#!/bin/bash\nset -euo pipefail\n'
    emit_export MASTER_ADDR "$master_addr"
    emit_export MASTER_PORT "$master_port"
    emit_export WORLD_SIZE "$world_size"
    emit_export RANK "$rank"
    emit_export LOCAL_RANK 0
    emit_export OMP_NUM_THREADS 2
    emit_export OPENBLAS_NUM_THREADS 1
    emit_export MKL_NUM_THREADS 1
    emit_export NUMEXPR_NUM_THREADS 1
    emit_export PYTORCH_CUDA_ALLOC_CONF expandable_segments:True
    emit_export NCCL_IB_DISABLE 0
    emit_export RANK_LOG "$rank_log"
    printf 'exec >"$RANK_LOG" 2>&1\n'
    printf 'test "$("%s/bin/python" -c '\''import torch; print(torch.cuda.device_count())'\'')" -eq 1\n' "$PYTHON_ROOT"
    printf 'test -n "$(ls -A /sys/class/infiniband 2>/dev/null)"\n'
    printf 'exec %q %q ' \
      "$PYTHON_ROOT/bin/python" \
      "$CODE_ROOT/train/train_ettr_progressive_coupling.py"
    printf '%q ' \
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
    printf '\n'
  } >"$staged"
  chmod 400 "$staged"
done

for rank in "${!jobs[@]}"; do
  job=${jobs[$rank]}
  staged="$launcher_root/rank-$(printf '%03d' "$rank").command"
  command="$CONTROL_ROOT/commands/$job.sh"
  digest="$CONTROL_ROOT/commands/$job.sha256"
  install -m 400 "$staged" "$command"
  digest_tmp=$(mktemp "$CONTROL_ROOT/commands/.${job}.XXXXXX")
  printf '%s\n' "$(sha256sum "$command" | awk '{print $1}')" >"$digest_tmp"
  chmod 400 "$digest_tmp"
  mv -n "$digest_tmp" "$digest"
  test -f "$digest"
done

printf \
  'dispatched ETTR coupling world=%s groups=%s output=%s commit=%s\n' \
  "$world_size" "$ALLOCATION_GROUPS" "$OUTDIR" "$SOURCE_COMMIT"
