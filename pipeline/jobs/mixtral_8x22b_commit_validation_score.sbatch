#!/bin/bash
#SBATCH --job-name=mixtral-commit-score
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=logs/mixtral_8x22b_commit_score_%j.out
#SBATCH --error=logs/mixtral_8x22b_commit_score_%j.err

set -euo pipefail

required=(
  PYTHON RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE ASSESSORS CANDIDATE_ROOT MODEL
  COMMIT_ROOT OUTPUT SANDBOX_RECEIPT
)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
[[ -x "$PYTHON" ]]
[[ -d "$RUNTIME" && ! -L "$RUNTIME" ]]
[[ -d "$CANDIDATE_ROOT" && ! -L "$CANDIDATE_ROOT" ]]
[[ -d "$COMMIT_ROOT" && ! -L "$COMMIT_ROOT" ]]
for path in "$SOURCE" "$ASSESSORS" "$MODEL" \
  "$COMMIT_ROOT/candidates.jsonl" "$COMMIT_ROOT/selections.jsonl" \
  "$COMMIT_ROOT/report.json"; do
  [[ -f "$path" && ! -L "$path" ]]
done
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]]
[[ ! -e "$SANDBOX_RECEIPT" && ! -L "$SANDBOX_RECEIPT" ]]

source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
q36_verify_runtime
q36_export_pythonpath
mkdir -p "$(dirname "$OUTPUT")" logs
export PATH=/apps/slurm/current/bin:/usr/bin:/bin
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$SLURM_TMPDIR"

"$PYTHON" -P -s -B "$RUNTIME/pipeline/score_mixtral_8x22b_commit_validation.py" \
  --source "$SOURCE" \
  --assessors "$ASSESSORS" \
  --candidate-root "$CANDIDATE_ROOT" \
  --model "$MODEL" \
  --commit-candidates "$COMMIT_ROOT/candidates.jsonl" \
  --selections "$COMMIT_ROOT/selections.jsonl" \
  --application-report "$COMMIT_ROOT/report.json" \
  --sandbox-receipt "$SANDBOX_RECEIPT" \
  --output "$OUTPUT"

[[ -f "$OUTPUT" && ! -L "$OUTPUT" ]]
[[ -f "$SANDBOX_RECEIPT" && ! -L "$SANDBOX_RECEIPT" ]]
chmod a-w "$OUTPUT" "$SANDBOX_RECEIPT"
