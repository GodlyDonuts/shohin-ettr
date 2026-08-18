#!/bin/bash
#SBATCH --job-name=mixtral-tp4-matched-eval
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc32,evc33,evc34,evc37,evc38,evc50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/mixtral_8x22b_multinode_tp_eval_%j.out
#SBATCH --error=logs/mixtral_8x22b_multinode_tp_eval_%j.err

set -euo pipefail

required=(
  PYTHON RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_MANIFEST
  MODEL_MANIFEST_SHA256 MECHANICS_REPORT REVISION_CHECKPOINT SOURCE
  DRAFT_CANDIDATES RUN_ROOT WORLD_RANK
)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
[[ "$WORLD_RANK" =~ ^[0-3]$ ]]
expected_rows=${EXPECTED_ROWS:-256}
shard_count=${SHARD_COUNT:-4}
group_index=${SHARD_GROUP_INDEX:-0}
group_count=${SHARD_GROUP_COUNT:-1}
case "$expected_rows:$shard_count:$group_index:$group_count" in
  256:4:0:1|1023:16:[0-3]:4) ;;
  *) printf 'evaluation geometry differs\n' >&2; exit 2 ;;
esac
[[ -x "$PYTHON" ]]
[[ -d "$RUNTIME" && ! -L "$RUNTIME" ]]
[[ -d "$MODEL_ROOT" && ! -L "$MODEL_ROOT" ]]
for path in "$MODEL_MANIFEST" "$MECHANICS_REPORT" "$REVISION_CHECKPOINT" "$SOURCE"; do
  [[ -f "$path" && ! -L "$path" ]]
done
IFS=: read -r -a draft_paths <<< "$DRAFT_CANDIDATES"
[[ ${#draft_paths[@]} -eq 4 ]]
draft_args=()
for path in "${draft_paths[@]}"; do
  [[ -f "$path" && ! -L "$path" ]]
  draft_args+=(--draft-candidates "$path")
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

coord="$RUN_ROOT/coordination/group_${group_index}"
[[ -d "$coord" && ! -L "$coord" ]]
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

output="$RUN_ROOT/candidates"
"$PYTHON" -P -s -B -m torch.distributed.run \
  --nnodes=4 \
  --nproc_per_node=1 \
  --node_rank="$WORLD_RANK" \
  --master_addr="$master_node" \
  --master_port="$master_port" \
  "$RUNTIME/train/hf_mixtral_8x22b_multinode_tp_evaluate_matched.py" \
  --model-root "$MODEL_ROOT" \
  --model-manifest "$MODEL_MANIFEST" \
  --expected-model-manifest-sha256 "$MODEL_MANIFEST_SHA256" \
  --mechanics-report "$MECHANICS_REPORT" \
  --revision-checkpoint "$REVISION_CHECKPOINT" \
  --source "$SOURCE" \
  "${draft_args[@]}" \
  --output-root "$output" \
  --expected-rows "$expected_rows" \
  --shard-count "$shard_count" \
  --shard-group-index "$group_index" \
  --shard-group-count "$group_count"

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
  shards_per_group=$((shard_count / group_count))
  first_shard=$((group_index * shards_per_group))
  final_shard=$((first_shard + shards_per_group - 1))
  for arm in unchanged self_refinement revision; do
    for shard_index in $(seq "$first_shard" "$final_shard"); do
      printf -v shard '%02d' "$shard_index"
      [[ -f "$output/$arm/shard_$shard/candidates.jsonl" ]]
      [[ -f "$output/$arm/shard_$shard/report.json" ]]
      find "$output/$arm/shard_$shard" -type f -exec chmod a-w {} +
      find "$output/$arm/shard_$shard" -type d -exec chmod a-w {} +
    done
  done
fi
