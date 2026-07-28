#!/bin/bash
# Launch a bounded multi-node DDP transport canary inside an already-running
# Shohin reservation. The canary writes only to its fresh isolated output.

set -euo pipefail

ALLOCATION_JOB_ID=${ALLOCATION_JOB_ID:?set the running reservation job ID}
CODE_ROOT=${CODE_ROOT:?set the immutable shared source checkout}
SOURCE_COMMIT=${SOURCE_COMMIT:?set the exact source commit}
OUTDIR=${OUTDIR:?set a fresh isolated canary output directory}
NODES=${NODES:?set the exact reserved node count}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
STEPS=${STEPS:-300}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-1}
PYTHON_ROOT=${PYTHON_ROOT:-/lustre/fs1/home/sa305415/shohin/miniforge3}
SHARD_ROOT=${SHARD_ROOT:-/lustre/fs1/home/sa305415/shohin/artifacts/shards}

case "$ALLOCATION_JOB_ID:$NODES:$GPUS_PER_NODE:$STEPS:$BATCH_SIZE:$GRAD_ACCUM" in
  *[!0-9:]* | *::* | :* | *:) echo "integer launch settings differ" >&2; exit 2 ;;
esac
if [[ ! "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "source commit differs" >&2
  exit 2
fi
for path in "$CODE_ROOT" "$OUTDIR" "$PYTHON_ROOT" "$SHARD_ROOT"; do
  if [[ "$path" != /* ]]; then
    echo "all canary paths must be absolute: $path" >&2
    exit 2
  fi
done
if [[ -e "$OUTDIR" || -L "$OUTDIR" ]]; then
  echo "refusing existing canary output: $OUTDIR" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_ROOT/bin/torchrun" ]]; then
  echo "torchrun is unavailable" >&2
  exit 2
fi

state=$(squeue -h -j "$ALLOCATION_JOB_ID" -o "%T")
name=$(squeue -h -j "$ALLOCATION_JOB_ID" -o "%j")
allocated_nodes=$(squeue -h -j "$ALLOCATION_JOB_ID" -o "%D")
node_list=$(squeue -h -j "$ALLOCATION_JOB_ID" -o "%N")
if [[ "$state" != "RUNNING" || "$name" != shohin-*h100-* ]]; then
  echo "requested job is not a running Shohin H100 reservation" >&2
  exit 2
fi
if [[ "$allocated_nodes" != "$NODES" || -z "$node_list" ]]; then
  echo "reservation node geometry differs" >&2
  exit 2
fi

cd "$CODE_ROOT"
if [[ "$(git rev-parse HEAD)" != "$SOURCE_COMMIT" ]]; then
  echo "source checkout commit differs" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "source checkout is not clean" >&2
  exit 2
fi
for shard in finemath4 openwebmath code_python finemath3; do
  if [[ ! -s "$SHARD_ROOT/$shard/manifest.json" ]]; then
    echo "missing admitted canary shard manifest: $shard" >&2
    exit 2
  fi
done

master_addr=$(scontrol show hostnames "$node_list" | head -n 1)
master_port=$((20000 + ALLOCATION_JOB_ID % 20000))
world_size=$((NODES * GPUS_PER_NODE))
mkdir -m 700 "$OUTDIR"

export CODE_ROOT OUTDIR NODES GPUS_PER_NODE STEPS BATCH_SIZE GRAD_ACCUM
export PYTHON_ROOT SHARD_ROOT master_addr master_port
export WORLD_SIZE="$world_size"
export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

printf \
  'canary job=%s nodes=%s gpus_per_node=%s world=%s master=%s commit=%s\n' \
  "$ALLOCATION_JOB_ID" "$NODES" "$GPUS_PER_NODE" "$world_size" \
  "$master_addr" "$SOURCE_COMMIT"

srun \
  --jobid="$ALLOCATION_JOB_ID" \
  --overlap \
  --nodes="$NODES" \
  --ntasks="$NODES" \
  --ntasks-per-node=1 \
  --cpus-per-task=4 \
  --gpus-per-node="$GPUS_PER_NODE" \
  --kill-on-bad-exit=1 \
  bash -lc '
    set -euo pipefail
    test "$(nvidia-smi -L | wc -l)" -eq "$GPUS_PER_NODE"
    test -n "$(ls -A /sys/class/infiniband 2>/dev/null)"
    "$PYTHON_ROOT/bin/torchrun" \
      --nnodes="$NODES" \
      --nproc_per_node="$GPUS_PER_NODE" \
      --node_rank="$SLURM_NODEID" \
      --rdzv_backend=c10d \
      --rdzv_endpoint="$master_addr:$master_port" \
      --rdzv_id="shohin-$SLURM_JOB_ID-ddp-canary" \
      "$CODE_ROOT/train/train.py" \
      --size shohin \
      --shard-dirs \
        "$SHARD_ROOT/finemath4" \
        "$SHARD_ROOT/openwebmath" \
        "$SHARD_ROOT/code_python" \
        "$SHARD_ROOT/finemath3" \
      --steps "$STEPS" \
      --batch-size "$BATCH_SIZE" \
      --grad-accum "$GRAD_ACCUM" \
      --warmup 50 \
      --log-every 10 \
      --ckpt-every 0 \
      --data-seed 20260728 \
      --out "$OUTDIR" \
      --compile
  '

test -s "$OUTDIR/log_r0.jsonl"
test -s "$OUTDIR/ckpt_final.pt"
sha256sum "$OUTDIR/log_r0.jsonl" "$OUTDIR/ckpt_final.pt" \
  > "$OUTDIR/SHA256SUMS"
chmod 400 "$OUTDIR/log_r0.jsonl" "$OUTDIR/ckpt_final.pt" "$OUTDIR/SHA256SUMS"
printf 'canary_complete output=%s\n' "$OUTDIR"
