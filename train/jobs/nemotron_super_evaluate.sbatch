#!/bin/bash
#SBATCH --job-name=nemotron-super-eval
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:nvidia_h100_pcie:2
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc33,evc34,evc37,evc38,evc50
#SBATCH --output=logs/nemotron_super_eval_%A_%a.out
#SBATCH --error=logs/nemotron_super_eval_%A_%a.err

set -euo pipefail

required=(
  PYTHON RUNTIME MODEL_ROOT MODEL_MANIFEST OVERLAY_ROOT CAUSAL_CONV_ROOT
  MECHANICS_REPORT ARM SOURCE DRAFT_CANDIDATES OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
[[ "$ARM" == unchanged || "$ARM" == self_refinement || "$ARM" == revision ]]
[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-3]$ ]]
[[ "${SLURM_GPUS_ON_NODE:-}" == "2" ]]
[[ -x "$PYTHON" ]]
for path in "$RUNTIME" "$MODEL_ROOT" "$OVERLAY_ROOT" "$CAUSAL_CONV_ROOT"; do
  [[ -d "$path" && ! -L "$path" ]]
done
for path in "$MODEL_MANIFEST" "$MECHANICS_REPORT" "$SOURCE"; do
  [[ -f "$path" && ! -L "$path" ]]
done
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
checkpoint_args=()
if [[ "$ARM" == revision ]]; then
  [[ -n "${REVISION_CHECKPOINT:-}" ]]
  [[ -f "$REVISION_CHECKPOINT" && ! -L "$REVISION_CHECKPOINT" ]]
  checkpoint_args=(--revision-checkpoint "$REVISION_CHECKPOINT")
elif [[ -n "${REVISION_CHECKPOINT:-}" ]]; then
  printf 'revision checkpoint supplied to control arm\n' >&2
  exit 2
fi
draft_args=()
if [[ "$ARM" == unchanged ]]; then
  [[ "$DRAFT_CANDIDATES" == none ]]
else
  IFS=: read -r -a draft_paths <<< "$DRAFT_CANDIDATES"
  [[ ${#draft_paths[@]} -eq 4 ]]
  for path in "${draft_paths[@]}"; do
    [[ -f "$path" && ! -L "$path" ]]
    draft_args+=(--draft-candidates "$path")
  done
fi

shard=$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
output_dir="$OUTPUT_ROOT/$ARM/shard_$shard"
candidates="$output_dir/candidates.jsonl"
report="$output_dir/report.json"
[[ ! -e "$candidates" && ! -L "$candidates" ]]
[[ ! -e "$report" && ! -L "$report" ]]
mkdir -p "$output_dir" logs

export PATH=/apps/slurm/current/bin:/usr/bin:/bin
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline:$OVERLAY_ROOT:$CAUSAL_CONV_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$SLURM_TMPDIR/hf" XDG_CACHE_HOME="$SLURM_TMPDIR/cache"
export TMPDIR="$SLURM_TMPDIR" TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
q36_export_nemotron_cuda_toolchain

"$PYTHON" -P -s -B "$RUNTIME/train/hf_nemotron_super_evaluate.py" \
  --arm "$ARM" \
  --model-root "$MODEL_ROOT" \
  --model-manifest "$MODEL_MANIFEST" \
  --mechanics-report "$MECHANICS_REPORT" \
  "${checkpoint_args[@]}" \
  --source "$SOURCE" \
  "${draft_args[@]}" \
  --candidates-output "$candidates" \
  --report "$report" \
  --shard-index "$SLURM_ARRAY_TASK_ID"

[[ -f "$candidates" && ! -L "$candidates" ]]
[[ -f "$report" && ! -L "$report" ]]
chmod a-w "$candidates" "$report" "$output_dir"
