#!/bin/bash
#SBATCH --job-name=nemotron-ultra-eval
#SBATCH --partition=highgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:8
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/nemotron_ultra_eval_%A_%a.out
#SBATCH --error=logs/nemotron_ultra_eval_%A_%a.err

set -euo pipefail

required=(
  PYTHON RUNTIME MODEL_ROOT MODEL_MANIFEST MODEL_MANIFEST_SHA256 OVERLAY_ROOT
  CAUSAL_CONV_ROOT MECHANICS_REPORT ARM SOURCE DRAFT_CANDIDATES OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
[[ "$ARM" == unchanged || "$ARM" == self_refinement || "$ARM" == revision ]]
[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-3]$ ]]
[[ "${SLURM_GPUS_ON_NODE:-}" == "8" ]]
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
  if [[ -n "${TRANSFER_REPORT:-}" && -z "${REVISION_TRAINING_REPORT:-}" ]]; then
    [[ -f "$TRANSFER_REPORT" && ! -L "$TRANSFER_REPORT" ]]
    checkpoint_args=(
      --revision-checkpoint "$REVISION_CHECKPOINT"
      --transfer-report "$TRANSFER_REPORT"
    )
  elif [[ -n "${REVISION_TRAINING_REPORT:-}" && -z "${TRANSFER_REPORT:-}" ]]; then
    [[ -f "$REVISION_TRAINING_REPORT" && ! -L "$REVISION_TRAINING_REPORT" ]]
    checkpoint_args=(
      --revision-checkpoint "$REVISION_CHECKPOINT"
      --revision-training-report "$REVISION_TRAINING_REPORT"
    )
  else
    printf 'Ultra revision requires exactly one checkpoint report\n' >&2
    exit 2
  fi
elif [[ -n "${REVISION_CHECKPOINT:-}" || -n "${TRANSFER_REPORT:-}" || -n "${REVISION_TRAINING_REPORT:-}" ]]; then
  printf 'Ultra revision inputs supplied to a control arm\n' >&2
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

"$PYTHON" -P -s -B "$RUNTIME/train/hf_nemotron_ultra_evaluate.py" \
  --arm "$ARM" \
  --model-root "$MODEL_ROOT" \
  --model-manifest "$MODEL_MANIFEST" \
  --expected-model-manifest-sha256 "$MODEL_MANIFEST_SHA256" \
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
