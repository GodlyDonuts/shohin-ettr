#!/bin/bash
# Run one bounded ETTR-v3 training rung and its paired development evaluation
# inside an existing multi-node H100 reservation.

set -euo pipefail

ALLOCATION_JOB_ID=${ALLOCATION_JOB_ID:?set the running reservation job ID}
CODE_ROOT=${CODE_ROOT:?set the immutable shared source root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set the exact private source commit}
RELEASE_ROOT=${RELEASE_ROOT:?set the immutable ETTR release root}
RELEASE_SHA256=${RELEASE_SHA256:?set the release.json SHA-256}
DATA_ROOT=${DATA_ROOT:?set the immutable ETTR shard root}
TOKENIZER=${TOKENIZER:?set the immutable tokenizer path}
PROTECTED_CHECKPOINT=${PROTECTED_CHECKPOINT:?set the protected 300k checkpoint}
OUTDIR=${OUTDIR:?set a fresh isolated pilot output directory}
NODELIST=${NODELIST:?set the healthy reserved nodes, comma separated}
NODES=${NODES:?set the exact selected node count}
START_UPDATE=${START_UPDATE:-0}
TARGET_UPDATE=${TARGET_UPDATE:-100}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
ACCUMULATION=${ACCUMULATION:-1}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-100}
LOG_EVERY=${LOG_EVERY:-10}
MAX_EVAL_BATCHES=${MAX_EVAL_BATCHES:-64}
ARCHITECTURE_SEED=${ARCHITECTURE_SEED:-2026072801}
DATA_SEED=${DATA_SEED:-2026072802}
TOTAL_UPDATES=${TOTAL_UPDATES:-300000}
WARMUP_UPDATES=${WARMUP_UPDATES:-2000}
FREEZE_BASE=${FREEZE_BASE:-1}
HARD_TRANSACTIONS=${HARD_TRANSACTIONS:-1}
NLL_GRADIENT_CAP=${NLL_GRADIENT_CAP:-}
COMPILE_MODE=${COMPILE_MODE:-default}
PYTHON_ROOT=${PYTHON_ROOT:-/lustre/fs1/home/sa305415/shohin/miniforge3}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}
RESUME_SHA256=${RESUME_SHA256:-}

integer_contract="$ALLOCATION_JOB_ID:$NODES:$GPUS_PER_NODE:$START_UPDATE"
integer_contract+=":$TARGET_UPDATE:$ACCUMULATION:$CHECKPOINT_EVERY:$LOG_EVERY"
integer_contract+=":$MAX_EVAL_BATCHES:$ARCHITECTURE_SEED:$DATA_SEED"
integer_contract+=":$TOTAL_UPDATES:$WARMUP_UPDATES:$FREEZE_BASE"
integer_contract+=":$HARD_TRANSACTIONS"
case "$integer_contract" in
  *[!0-9:]* | *::* | :* | *:)
    echo "integer launch settings differ" >&2
    exit 2
    ;;
esac
if (( TARGET_UPDATE <= START_UPDATE || TARGET_UPDATE > TOTAL_UPDATES )); then
  echo "ETTR update range differs" >&2
  exit 2
fi
UPDATES=$((TARGET_UPDATE - START_UPDATE))
if [[ ! "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
  || [[ ! "$RELEASE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "source or release identity differs" >&2
  exit 2
fi
if [[ "$FREEZE_BASE" != 0 && "$FREEZE_BASE" != 1 ]]; then
  echo "freeze-base flag differs" >&2
  exit 2
fi
if [[ "$HARD_TRANSACTIONS" != 0 && "$HARD_TRANSACTIONS" != 1 ]]; then
  echo "transaction mode differs" >&2
  exit 2
fi
if [[ -n "$NLL_GRADIENT_CAP" \
  && ( ! "$NLL_GRADIENT_CAP" =~ ^[0-9]+([.][0-9]+)?$ \
    || "$HARD_TRANSACTIONS" != 1 ) ]]; then
  echo "NLL gradient cap differs" >&2
  exit 2
fi
if [[ "$COMPILE_MODE" != eager \
  && "$COMPILE_MODE" != default \
  && "$COMPILE_MODE" != reduce-overhead \
  && "$COMPILE_MODE" != max-autotune \
  && "$COMPILE_MODE" != max-autotune-no-cudagraphs ]]; then
  echo "compile mode differs" >&2
  exit 2
fi
if [[ "$NODELIST" == *" "* || "$NODELIST" == *"["* || "$NODELIST" == *"]"* ]]; then
  echo "node list must be a concrete comma-separated node list" >&2
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
    echo "all ETTR pilot paths must be absolute: $path" >&2
    exit 2
  fi
done
if [[ -e "$OUTDIR" || -L "$OUTDIR" ]]; then
  echo "refusing existing ETTR pilot output: $OUTDIR" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_ROOT/bin/torchrun" ]]; then
  echo "torchrun is unavailable" >&2
  exit 2
fi
if [[ "$START_UPDATE" == 0 ]]; then
  if [[ -n "$RESUME_CHECKPOINT" || -n "$RESUME_SHA256" ]]; then
    echo "initial ETTR rung may not provide resume state" >&2
    exit 2
  fi
else
  if [[ "$RESUME_CHECKPOINT" != /* \
    || ! "$RESUME_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "continuation ETTR rung requires exact resume state" >&2
    exit 2
  fi
fi

state=$(squeue -h -j "$ALLOCATION_JOB_ID" -o "%T")
name=$(squeue -h -j "$ALLOCATION_JOB_ID" -o "%j")
allocated_nodes=$(squeue -h -j "$ALLOCATION_JOB_ID" -o "%N")
if [[ "$state" != "RUNNING" || "$name" != shohin-*h100-* ]]; then
  echo "requested job is not a running Shohin H100 reservation" >&2
  exit 2
fi
if ! scontrol show hostnames "$allocated_nodes" \
  | sort -u > "/tmp/shohin_ettr_alloc_nodes.$$"; then
  echo "allocated node list cannot be expanded" >&2
  exit 2
fi
trap 'rm -f "/tmp/shohin_ettr_alloc_nodes.$$" "/tmp/shohin_ettr_selected_nodes.$$"' EXIT
tr ',' '\n' <<< "$NODELIST" \
  | sort -u > "/tmp/shohin_ettr_selected_nodes.$$"
if [[ "$(wc -l < "/tmp/shohin_ettr_selected_nodes.$$")" != "$NODES" ]] \
  || comm \
    -23 \
    "/tmp/shohin_ettr_selected_nodes.$$" \
    "/tmp/shohin_ettr_alloc_nodes.$$" \
    | grep -q .; then
  echo "selected node geometry differs from reservation" >&2
  exit 2
fi

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

master_addr=$(tr ',' '\n' <<< "$NODELIST" | head -n 1)
master_port=$((20000 + ALLOCATION_JOB_ID % 20000))
world_size=$((NODES * GPUS_PER_NODE))
mkdir -m 700 "$OUTDIR"

export ALLOCATION_JOB_ID CODE_ROOT SOURCE_COMMIT RELEASE_ROOT RELEASE_SHA256
export DATA_ROOT TOKENIZER PROTECTED_CHECKPOINT OUTDIR NODES NODELIST
export GPUS_PER_NODE START_UPDATE TARGET_UPDATE UPDATES ACCUMULATION
export CHECKPOINT_EVERY LOG_EVERY MAX_EVAL_BATCHES ARCHITECTURE_SEED DATA_SEED
export TOTAL_UPDATES WARMUP_UPDATES FREEZE_BASE HARD_TRANSACTIONS
export NLL_GRADIENT_CAP
export COMPILE_MODE PYTHON_ROOT
export RESUME_CHECKPOINT RESUME_SHA256 master_addr master_port world_size
export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=0

printf \
  'ettr_pilot job=%s nodes=%s gpus_per_node=%s world=%s updates=%s..%s freeze_base=%s commit=%s\n' \
  "$ALLOCATION_JOB_ID" "$NODES" "$GPUS_PER_NODE" "$world_size" \
  "$START_UPDATE" "$TARGET_UPDATE" "$FREEZE_BASE" "$SOURCE_COMMIT"

srun \
  --jobid="$ALLOCATION_JOB_ID" \
  --overlap \
  --nodes="$NODES" \
  --nodelist="$NODELIST" \
  --ntasks="$NODES" \
  --ntasks-per-node=1 \
  --cpus-per-task=4 \
  --gpus-per-node="$GPUS_PER_NODE" \
  --kill-on-bad-exit=1 \
  bash -lc '
    set -euo pipefail
    test "$(nvidia-smi -L | wc -l)" -ge "$GPUS_PER_NODE"
    test -n "$(ls -A /sys/class/infiniband 2>/dev/null)"
    freeze_args=()
    resume_args=()
    compile_args=()
    transaction_args=()
    if [[ "$FREEZE_BASE" == 1 ]]; then
      freeze_args+=(--freeze-base)
    fi
    if [[ "$COMPILE_MODE" != eager ]]; then
      compile_args+=(--compile-mode "$COMPILE_MODE")
    fi
    if [[ "$HARD_TRANSACTIONS" == 0 ]]; then
      transaction_args+=(--soft-transactions)
    fi
    if [[ -n "$NLL_GRADIENT_CAP" ]]; then
      transaction_args+=(--nll-gradient-cap "$NLL_GRADIENT_CAP")
    fi
    if [[ "$START_UPDATE" != 0 ]]; then
      resume_args+=(
        --resume-checkpoint "$RESUME_CHECKPOINT"
        --resume-sha256 "$RESUME_SHA256"
      )
    fi
    "$PYTHON_ROOT/bin/torchrun" \
      --nnodes="$NODES" \
      --nproc_per_node="$GPUS_PER_NODE" \
      --node_rank="$SLURM_NODEID" \
      --rdzv_backend=c10d \
      --rdzv_endpoint="$master_addr:$master_port" \
      --rdzv_id="shohin-$SLURM_JOB_ID-ettr-$TARGET_UPDATE" \
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
      "${compile_args[@]}" \
      "${transaction_args[@]}" \
      "${freeze_args[@]}" \
      "${resume_args[@]}"
  '

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
  echo "ETTR checkpoint or run-contract identity differs" >&2
  exit 2
fi

srun \
  --jobid="$ALLOCATION_JOB_ID" \
  --overlap \
  --nodes=1 \
  --nodelist="$master_addr" \
  --ntasks=1 \
  --cpus-per-task=4 \
  --gpus-per-node="$GPUS_PER_NODE" \
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
  > "$OUTDIR/SHA256SUMS"
chmod 400 \
  "$checkpoint" \
  "$sidecar" \
  "$contract" \
  "$OUTDIR/development-evaluation.json" \
  "$OUTDIR/SHA256SUMS"
printf 'ettr_pilot_complete output=%s target_update=%s\n' \
  "$OUTDIR" \
  "$TARGET_UPDATE"
