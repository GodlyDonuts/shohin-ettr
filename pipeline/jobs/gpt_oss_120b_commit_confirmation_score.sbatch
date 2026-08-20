#!/bin/bash
#SBATCH --job-name=gpt-oss-confirm-score
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=logs/gpt_oss_confirm_score_%j.out
#SBATCH --error=logs/gpt_oss_confirm_score_%j.err

set -euo pipefail
for variable in PYTHON RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE EXPECTED_SOURCE_SHA256 ASSESSORS EXPECTED_ASSESSORS_SHA256 CANDIDATE_ROOT OUTPUT; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
q36_verify_runtime
for path in "$SOURCE" "$ASSESSORS"; do [[ -f "$path" && ! -L "$path" ]]; done
[[ -d "$CANDIDATE_ROOT" && ! -L "$CANDIDATE_ROOT" ]]
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]]
args=()
for arm in revision unchanged; do
  for shard in 00 01 02 03; do
    path="$CANDIDATE_ROOT/$arm/shard_$shard/candidates.jsonl"
    [[ -f "$path" && ! -L "$path" ]]
    args+=("--${arm}-candidates" "$path")
  done
done
mkdir -p "$(dirname "$OUTPUT")" logs
export PATH=/apps/slurm/current/bin:/usr/bin:/bin
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$SLURM_TMPDIR"
"$PYTHON" -P -s -B "$RUNTIME/pipeline/score_gpt_oss_120b_commit_confirmation.py" \
  --source "$SOURCE" --expected-source-sha256 "$EXPECTED_SOURCE_SHA256" \
  --assessors "$ASSESSORS" --expected-assessors-sha256 "$EXPECTED_ASSESSORS_SHA256" \
  "${args[@]}" --output "$OUTPUT"
test -f "$OUTPUT"
chmod a-w "$OUTPUT"
