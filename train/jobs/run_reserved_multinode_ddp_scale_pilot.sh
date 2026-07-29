#!/bin/bash
# Run a bounded full-stack multi-node DDP scale pilot inside an existing H100
# reservation. This pilot uses only previously admitted historical shards and
# writes to one fresh, isolated directory.

set -euo pipefail

ALLOCATION_JOB_ID=${ALLOCATION_JOB_ID:?set the running reservation job ID}
CODE_ROOT=${CODE_ROOT:?set the immutable shared source archive root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set the exact private source commit}
OUTDIR=${OUTDIR:?set a fresh isolated pilot output directory}
NODELIST=${NODELIST:?set the healthy reserved nodes, comma separated}
NODES=${NODES:?set the exact healthy reserved node count}
STEPS=${STEPS:-80}
BATCH_SIZE=${BATCH_SIZE:-16}
GRAD_ACCUM=${GRAD_ACCUM:-8}
LR_TOTAL_STEPS=${LR_TOTAL_STEPS:-300000}
WARMUP=${WARMUP:-2000}
PYTHON_ROOT=${PYTHON_ROOT:-/lustre/fs1/home/sa305415/shohin/miniforge3}
SHARD_ROOT=${SHARD_ROOT:-/lustre/fs1/home/sa305415/shohin/artifacts/shards}

case "$ALLOCATION_JOB_ID:$NODES:$STEPS:$BATCH_SIZE:$GRAD_ACCUM:$LR_TOTAL_STEPS:$WARMUP" in
  *[!0-9:]* | *::* | :* | *:) echo "integer launch settings differ" >&2; exit 2 ;;
esac
if [[ ! "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "source commit differs" >&2
  exit 2
fi
if [[ "$NODELIST" == *" "* || "$NODELIST" == *"["* || "$NODELIST" == *"]"* ]]; then
  echo "node list must be a concrete comma-separated node list" >&2
  exit 2
fi
for path in "$CODE_ROOT" "$OUTDIR" "$PYTHON_ROOT" "$SHARD_ROOT"; do
  if [[ "$path" != /* ]]; then
    echo "all pilot paths must be absolute: $path" >&2
    exit 2
  fi
done
if [[ -e "$OUTDIR" || -L "$OUTDIR" ]]; then
  echo "refusing existing scale-pilot output: $OUTDIR" >&2
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
if ! scontrol show hostnames "$allocated_nodes" | sort -u > /tmp/shohin_alloc_nodes.$$; then
  echo "allocated node list cannot be expanded" >&2
  exit 2
fi
trap 'rm -f /tmp/shohin_alloc_nodes.$$ /tmp/shohin_selected_nodes.$$' EXIT
tr ',' '\n' <<< "$NODELIST" | sort -u > /tmp/shohin_selected_nodes.$$
if [[ "$(wc -l < /tmp/shohin_selected_nodes.$$)" != "$NODES" ]] \
  || comm -23 /tmp/shohin_selected_nodes.$$ /tmp/shohin_alloc_nodes.$$ | grep -q .; then
  echo "selected node geometry differs from reservation" >&2
  exit 2
fi

cd "$CODE_ROOT"
if [[ ! -r SOURCE_COMMIT || "$(tr -d '\r\n' < SOURCE_COMMIT)" != "$SOURCE_COMMIT" ]]; then
  echo "source archive commit differs" >&2
  exit 2
fi
if [[ ! -r SHA256SUMS || -n "$(sha256sum -c SHA256SUMS 2>&1 | grep -v ': OK$')" ]]; then
  echo "source archive digest differs" >&2
  exit 2
fi
for shard in finemath4 openwebmath code_python finemath3; do
  if [[ ! -s "$SHARD_ROOT/$shard/manifest.json" ]]; then
    echo "missing admitted scale-pilot shard manifest: $shard" >&2
    exit 2
  fi
done

master_addr=$(tr ',' '\n' <<< "$NODELIST" | head -n 1)
master_port=$((20000 + ALLOCATION_JOB_ID % 20000))
mkdir -m 700 "$OUTDIR"

export CODE_ROOT OUTDIR NODES NODELIST STEPS BATCH_SIZE GRAD_ACCUM
export LR_TOTAL_STEPS WARMUP PYTHON_ROOT SHARD_ROOT master_addr master_port
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

printf \
  'scale_pilot job=%s nodes=%s nodelist=%s bs=%s accum=%s commit=%s\n' \
  "$ALLOCATION_JOB_ID" "$NODES" "$NODELIST" "$BATCH_SIZE" "$GRAD_ACCUM" \
  "$SOURCE_COMMIT"

srun \
  --jobid="$ALLOCATION_JOB_ID" \
  --overlap \
  --nodes="$NODES" \
  --nodelist="$NODELIST" \
  --ntasks="$NODES" \
  --ntasks-per-node=1 \
  --cpus-per-task=4 \
  --gpus-per-node=1 \
  --kill-on-bad-exit=1 \
  bash -lc '
    set -euo pipefail
    test "$(nvidia-smi -L | wc -l)" -ge 1
    test -n "$(ls -A /sys/class/infiniband 2>/dev/null)"
    "$PYTHON_ROOT/bin/python" -c "import torch; assert torch.cuda.is_available()"
    "$PYTHON_ROOT/bin/torchrun" \
      --nnodes="$NODES" \
      --nproc_per_node=1 \
      --node_rank="$SLURM_NODEID" \
      --rdzv_backend=c10d \
      --rdzv_endpoint="$master_addr:$master_port" \
      --rdzv_id="shohin-$SLURM_JOB_ID-scale-pilot" \
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
      --lr-total-steps "$LR_TOTAL_STEPS" \
      --warmup "$WARMUP" \
      --log-every 10 \
      --ckpt-every 0 \
      --data-seed 20260729 \
      --out "$OUTDIR" \
      --compile
  '

test -s "$OUTDIR/log_r0.jsonl"
test -s "$OUTDIR/ckpt_final.pt"
sha256sum "$OUTDIR/log_r0.jsonl" "$OUTDIR/ckpt_final.pt" > "$OUTDIR/SHA256SUMS"
chmod 400 "$OUTDIR/log_r0.jsonl" "$OUTDIR/ckpt_final.pt" "$OUTDIR/SHA256SUMS"
printf 'scale_pilot_complete output=%s\n' "$OUTDIR"
