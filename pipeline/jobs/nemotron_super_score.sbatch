#!/bin/bash
#SBATCH --job-name=nemotron-super-score
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=logs/nemotron_super_score_%j.out
#SBATCH --error=logs/nemotron_super_score_%j.err

set -euo pipefail

required=(PYTHON RUNTIME ASSESSORS CANDIDATE_ROOT OUTPUT SANDBOX_RECEIPT)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
[[ -x "$PYTHON" ]]
[[ -d "$RUNTIME" && ! -L "$RUNTIME" ]]
[[ -d "$CANDIDATE_ROOT" && ! -L "$CANDIDATE_ROOT" ]]
[[ -f "$ASSESSORS" && ! -L "$ASSESSORS" ]]
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]]
[[ ! -e "$SANDBOX_RECEIPT" && ! -L "$SANDBOX_RECEIPT" ]]
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
args=()
for arm in unchanged self_refinement revision; do
  option=${arm//_/-}
  for shard in 00 01 02 03; do
    path="$CANDIDATE_ROOT/$arm/shard_$shard/candidates.jsonl"
    [[ -f "$path" && ! -L "$path" ]]
    args+=("--${option}-candidates" "$path")
  done
done
mkdir -p "$(dirname "$OUTPUT")" logs

export PATH=/apps/slurm/current/bin:/usr/bin:/bin
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$SLURM_TMPDIR"
"$PYTHON" -P -s -B "$RUNTIME/pipeline/score_nemotron_super_screen.py" \
  --assessors "$ASSESSORS" \
  "${args[@]}" \
  --sandbox-receipt "$SANDBOX_RECEIPT" \
  --output "$OUTPUT"

[[ -f "$OUTPUT" && ! -L "$OUTPUT" ]]
[[ -f "$SANDBOX_RECEIPT" && ! -L "$SANDBOX_RECEIPT" ]]
chmod a-w "$OUTPUT" "$SANDBOX_RECEIPT"
