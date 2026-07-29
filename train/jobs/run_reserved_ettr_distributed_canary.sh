#!/bin/bash
# Run the synthetic multi-node ETTR gradient canary inside an existing H100
# reservation. No training shard or ETTR release is visible to this job step.

set -euo pipefail

ALLOCATION_JOB_ID=${ALLOCATION_JOB_ID:?set the running reservation job ID}
CODE_ROOT=${CODE_ROOT:?set the immutable shared source root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set the exact private source commit}
PROTECTED_CHECKPOINT=${PROTECTED_CHECKPOINT:?set the protected 300k checkpoint}
PROTECTED_CHECKPOINT_SHA256=${PROTECTED_CHECKPOINT_SHA256:?set its SHA-256}
OUTDIR=${OUTDIR:?set a fresh isolated canary output directory}
NODELIST=${NODELIST:?set the healthy reserved nodes, comma separated}
NODES=${NODES:?set the exact selected node count}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
COMPILE_MODE=${COMPILE_MODE:-default}
PYTHON_ROOT=${PYTHON_ROOT:-/lustre/fs1/home/sa305415/shohin/miniforge3}

case "$ALLOCATION_JOB_ID:$NODES:$GPUS_PER_NODE" in
  *[!0-9:]* | *::* | :* | *:)
    echo "integer canary settings differ" >&2
    exit 2
    ;;
esac
if [[ ! "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
  || [[ ! "$PROTECTED_CHECKPOINT_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "canary source or checkpoint identity differs" >&2
  exit 2
fi
if [[ "$COMPILE_MODE" != default \
  && "$COMPILE_MODE" != reduce-overhead \
  && "$COMPILE_MODE" != max-autotune ]]; then
  echo "canary compile mode differs" >&2
  exit 2
fi
if [[ "$NODELIST" == *" "* || "$NODELIST" == *"["* || "$NODELIST" == *"]"* ]]; then
  echo "node list must be a concrete comma-separated node list" >&2
  exit 2
fi
for path in "$CODE_ROOT" "$PROTECTED_CHECKPOINT" "$OUTDIR" "$PYTHON_ROOT"; do
  if [[ "$path" != /* ]]; then
    echo "all distributed-canary paths must be absolute: $path" >&2
    exit 2
  fi
done
if [[ -e "$OUTDIR" || -L "$OUTDIR" ]]; then
  echo "refusing existing distributed-canary output: $OUTDIR" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_ROOT/bin/torchrun" ]]; then
  echo "torchrun is unavailable" >&2
  exit 2
fi

state=$(squeue -h -j "$ALLOCATION_JOB_ID" -o "%T")
name=$(squeue -h -j "$ALLOCATION_JOB_ID" -o "%j")
allocated_nodes=$(squeue -h -j "$ALLOCATION_JOB_ID" -o "%N")
if [[ "$state" != "RUNNING" || "$name" != shohin-*h100-* ]]; then
  echo "requested job is not a running Shohin H100 reservation" >&2
  exit 2
fi
if ! scontrol show hostnames "$allocated_nodes" \
  | sort -u > "/tmp/shohin_ettr_canary_alloc_nodes.$$"; then
  echo "allocated node list cannot be expanded" >&2
  exit 2
fi
trap 'rm -f "/tmp/shohin_ettr_canary_alloc_nodes.$$" "/tmp/shohin_ettr_canary_selected_nodes.$$"' EXIT
tr ',' '\n' <<< "$NODELIST" \
  | sort -u > "/tmp/shohin_ettr_canary_selected_nodes.$$"
if [[ "$(wc -l < "/tmp/shohin_ettr_canary_selected_nodes.$$")" != "$NODES" ]] \
  || comm \
    -23 \
    "/tmp/shohin_ettr_canary_selected_nodes.$$" \
    "/tmp/shohin_ettr_canary_alloc_nodes.$$" \
    | grep -q .; then
  echo "selected canary geometry differs from reservation" >&2
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
if [[ ! -s "$PROTECTED_CHECKPOINT" \
  || "$(sha256sum "$PROTECTED_CHECKPOINT" | awk '{print $1}')" \
    != "$PROTECTED_CHECKPOINT_SHA256" ]]; then
  echo "protected checkpoint identity differs" >&2
  exit 2
fi

master_addr=$(tr ',' '\n' <<< "$NODELIST" | head -n 1)
master_port=$((30000 + ALLOCATION_JOB_ID % 20000))
world_size=$((NODES * GPUS_PER_NODE))
export CODE_ROOT SOURCE_COMMIT PROTECTED_CHECKPOINT
export PROTECTED_CHECKPOINT_SHA256 OUTDIR NODES GPUS_PER_NODE
export COMPILE_MODE PYTHON_ROOT master_addr master_port world_size
export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=0

printf \
  'ettr_distributed_canary job=%s nodes=%s gpus_per_node=%s world=%s commit=%s\n' \
  "$ALLOCATION_JOB_ID" "$NODES" "$GPUS_PER_NODE" "$world_size" "$SOURCE_COMMIT"

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
    "$PYTHON_ROOT/bin/torchrun" \
      --nnodes="$NODES" \
      --nproc_per_node="$GPUS_PER_NODE" \
      --node_rank="$SLURM_NODEID" \
      --rdzv_backend=c10d \
      --rdzv_endpoint="$master_addr:$master_port" \
      --rdzv_id="shohin-$SLURM_JOB_ID-ettr-distributed-canary" \
      "$CODE_ROOT/train/canary_ettr_distributed_h100.py" \
      --output "$OUTDIR" \
      --checkpoint "$PROTECTED_CHECKPOINT" \
      --checkpoint-sha256 "$PROTECTED_CHECKPOINT_SHA256" \
      --expected-step 300000 \
      --source-commit "$SOURCE_COMMIT" \
      --expected-world-size "$world_size" \
      --compile-mode "$COMPILE_MODE"
  '

test -s "$OUTDIR/report.json"
sha256sum "$OUTDIR/report.json" > "$OUTDIR/SHA256SUMS"
chmod 400 "$OUTDIR/report.json" "$OUTDIR/SHA256SUMS"
printf 'ettr_distributed_canary_complete output=%s\n' "$OUTDIR"
