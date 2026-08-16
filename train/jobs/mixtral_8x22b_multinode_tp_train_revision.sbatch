#!/bin/bash
#SBATCH --job-name=mixtral-tp4-revision
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc32,evc33,evc34,evc37,evc38,evc43,evc46,evc50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/mixtral_8x22b_multinode_tp_train_%j.out
#SBATCH --error=logs/mixtral_8x22b_multinode_tp_train_%j.err

set -euo pipefail

required=(
  PYTHON RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_MANIFEST
  MODEL_MANIFEST_SHA256 MECHANICS_REPORT DATA RUN_ROOT WORLD_RANK
)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
[[ "$WORLD_RANK" =~ ^[0-3]$ ]]
[[ -x "$PYTHON" ]]
[[ -d "$RUNTIME" && ! -L "$RUNTIME" ]]
[[ -d "$MODEL_ROOT" && ! -L "$MODEL_ROOT" ]]
for path in "$MODEL_MANIFEST" "$MECHANICS_REPORT" "$DATA"; do
  [[ -f "$path" && ! -L "$path" ]]
done
[[ -d "$RUN_ROOT" && ! -L "$RUN_ROOT" ]]
[[ "${SLURM_GPUS_ON_NODE:-}" == 1 ]]

source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
q36_verify_runtime
q36_export_pythonpath
mkdir -p logs

export PATH=/apps/slurm/current/bin:/usr/bin:/bin
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$SLURM_TMPDIR/hf" XDG_CACHE_HOME="$SLURM_TMPDIR/cache"
export TMPDIR="$SLURM_TMPDIR" TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NCCL_IB_DISABLE=0 TORCH_NCCL_ASYNC_ERROR_HANDLING=1

coord="$RUN_ROOT/coordination"
mkdir -p "$coord"
admitted="$coord/rank-${WORLD_RANK}.admitted"
[[ ! -e "$admitted" && ! -L "$admitted" ]]
temporary="$coord/.rank-${WORLD_RANK}.admitted.tmp.${SLURM_JOB_ID}"
printf 'rank=%s\njob=%s\nnode=%s\ngpu=%s\n' \
  "$WORLD_RANK" "$SLURM_JOB_ID" "$SLURMD_NODENAME" "${CUDA_VISIBLE_DEVICES:-}" \
  > "$temporary"
chmod 400 "$temporary"
mv "$temporary" "$admitted"

all_admitted() {
  [[ -s "$coord/rank-0.admitted" && -s "$coord/rank-1.admitted" \
    && -s "$coord/rank-2.admitted" && -s "$coord/rank-3.admitted" ]]
}
for _ in $(seq 1 21600); do
  all_admitted && break
  sleep 1
done
all_admitted
master_node=$(awk -F= '$1 == "node" {print $2}' "$coord/rank-0.admitted")
master_job=$(awk -F= '$1 == "job" {print $2}' "$coord/rank-0.admitted")
[[ "$master_node" == evc[0-9]* && "$master_job" =~ ^[0-9]+$ ]]
master_port=$((30000 + master_job % 20000))

output="$RUN_ROOT/training"
"$PYTHON" -P -s -B -m torch.distributed.run \
  --nnodes=4 \
  --nproc_per_node=1 \
  --node_rank="$WORLD_RANK" \
  --master_addr="$master_node" \
  --master_port="$master_port" \
  "$RUNTIME/train/hf_mixtral_8x22b_multinode_tp_train_revision.py" \
  --model-root "$MODEL_ROOT" \
  --model-manifest "$MODEL_MANIFEST" \
  --expected-model-manifest-sha256 "$MODEL_MANIFEST_SHA256" \
  --mechanics-report "$MECHANICS_REPORT" \
  --data "$DATA" \
  --output "$output"

completed="$coord/rank-${WORLD_RANK}.completed"
[[ ! -e "$completed" && ! -L "$completed" ]]
printf 'rank=%s\njob=%s\nnode=%s\ncompleted=1\n' \
  "$WORLD_RANK" "$SLURM_JOB_ID" "$SLURMD_NODENAME" > "$completed"
chmod 400 "$completed"
if [[ "$WORLD_RANK" == 0 ]]; then
  all_completed() {
    [[ -s "$coord/rank-0.completed" && -s "$coord/rank-1.completed" \
      && -s "$coord/rank-2.completed" && -s "$coord/rank-3.completed" ]]
  }
  for _ in $(seq 1 300); do
    all_completed && break
    sleep 1
  done
  all_completed
  checkpoint="$output/checkpoint_0000256.pt"
  report="$output/report.json"
  [[ -f "$checkpoint" && ! -L "$checkpoint" ]]
  [[ -f "$report" && ! -L "$report" ]]
  chmod a-w "$checkpoint" "$report" "$output"
fi
