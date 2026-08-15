#!/bin/bash
#SBATCH --job-name=q36-mtr-temporal-score
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_mtr_temporal_score_%j.out
#SBATCH --error=logs/q36_mtr_temporal_score_%j.err

set -euo pipefail

required=(
  RUNTIME RUNTIME_MANIFEST_SHA256 PYTHON ASSESSORS BASELINE_SCORE
  TEMPORAL_CANDIDATES EXPECTED_ROWS SHARD_COUNT OUTPUT SANDBOX_RECEIPT
)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
ARM=${ARM:-temporal_gate}
[[ "$ARM" == temporal_gate || "$ARM" == multi_trajectory_gate ]] || \
  q36_die "temporal score arm differs"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
q36_verify_runtime
q36_require_file "$ASSESSORS"
q36_require_file "$BASELINE_SCORE"
IFS=: read -r -a candidate_paths <<< "$TEMPORAL_CANDIDATES"
[[ ${#candidate_paths[@]} -eq SHARD_COUNT ]] || q36_die "temporal candidate geometry differs"
candidate_args=()
for path in "${candidate_paths[@]}"; do
  q36_require_file "$path"
  candidate_args+=(--temporal-candidates "$path")
done
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || q36_die "temporal score output exists"
[[ ! -e "$SANDBOX_RECEIPT" && ! -L "$SANDBOX_RECEIPT" ]] || \
  q36_die "temporal sandbox output exists"
mkdir -p "$(dirname "$OUTPUT")" logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
q36_export_pythonpath
"$PYTHON" "$RUNTIME/pipeline/score_q36_mtr_temporal_gate.py" \
  --assessors "$ASSESSORS" \
  --baseline-score "$BASELINE_SCORE" \
  --arm "$ARM" \
  "${candidate_args[@]}" \
  --expected-rows "$EXPECTED_ROWS" \
  --shard-count "$SHARD_COUNT" \
  --sandbox-receipt "$SANDBOX_RECEIPT" \
  --output "$OUTPUT"
test -f "$OUTPUT" && test -f "$SANDBOX_RECEIPT"
chmod a-w "$OUTPUT" "$SANDBOX_RECEIPT"
