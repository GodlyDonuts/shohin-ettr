#!/bin/bash
#SBATCH --job-name=score-upward-temporal
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/score_upward_temporal_%j.out
#SBATCH --error=logs/score_upward_temporal_%j.err

set -euo pipefail
for variable in HOST PYTHON RUNTIME RUNTIME_MANIFEST_SHA256 ASSESSORS EVALUATION_ROOT OUTPUT SANDBOX_RECEIPT; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
[[ "$HOST" == nemotron-super || "$HOST" == mixtral-8x22b ]]
[[ -f "$ASSESSORS" && ! -L "$ASSESSORS" ]]
[[ -d "$EVALUATION_ROOT" && ! -L "$EVALUATION_ROOT" ]]
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]]
[[ ! -e "$SANDBOX_RECEIPT" && ! -L "$SANDBOX_RECEIPT" ]]

source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
q36_verify_runtime
args=()
for arm in unchanged self_refinement owner aligned_revision temporal_gate; do
  for index in $(seq 0 15); do
    shard=$(printf '%02d' "$index")
    candidates=$EVALUATION_ROOT/$arm/shard_$shard/candidates.jsonl
    report=$EVALUATION_ROOT/$arm/shard_$shard/report.json
    [[ -f "$candidates" && ! -L "$candidates" ]]
    [[ -f "$report" && ! -L "$report" ]]
    args+=(--candidate "$arm" "$candidates")
    args+=(--evaluation-report "$arm" "$report")
  done
done
mkdir -p "$(dirname "$OUTPUT")" logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$RUNTIME/pipeline:$RUNTIME/train"
"$PYTHON" -P -s -B "$RUNTIME/pipeline/score_upward_moe_temporal_gate.py" \
  --host "$HOST" --assessors "$ASSESSORS" "${args[@]}" \
  --sandbox-receipt "$SANDBOX_RECEIPT" --output "$OUTPUT"
[[ -f "$OUTPUT" && ! -L "$OUTPUT" ]]
[[ -f "$SANDBOX_RECEIPT" && ! -L "$SANDBOX_RECEIPT" ]]
chmod a-w "$OUTPUT" "$SANDBOX_RECEIPT"
