#!/bin/bash
#SBATCH --job-name=mixtral-commit-apply
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=logs/mixtral_8x22b_commit_apply_%j.out
#SBATCH --error=logs/mixtral_8x22b_commit_apply_%j.err

set -euo pipefail

required=(
  PYTHON RUNTIME RUNTIME_MANIFEST_SHA256 MODEL SOURCE CANDIDATE_ROOT OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
[[ -x "$PYTHON" ]]
[[ -d "$RUNTIME" && ! -L "$RUNTIME" ]]
[[ -d "$CANDIDATE_ROOT" && ! -L "$CANDIDATE_ROOT" ]]
[[ -f "$MODEL" && ! -L "$MODEL" ]]
[[ -f "$SOURCE" && ! -L "$SOURCE" ]]
[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]]

source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
q36_verify_runtime
q36_export_pythonpath
mkdir -p logs
export PATH=/apps/slurm/current/bin:/usr/bin:/bin
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

"$PYTHON" -P -s -B "$RUNTIME/pipeline/train_apply_mixtral_8x22b_commit.py" apply \
  --model "$MODEL" \
  --source "$SOURCE" \
  --candidate-root "$CANDIDATE_ROOT" \
  --output "$OUTPUT_ROOT/candidates.jsonl" \
  --selections "$OUTPUT_ROOT/selections.jsonl" \
  --report "$OUTPUT_ROOT/report.json"

[[ -f "$OUTPUT_ROOT/candidates.jsonl" && ! -L "$OUTPUT_ROOT/candidates.jsonl" ]]
[[ -f "$OUTPUT_ROOT/selections.jsonl" && ! -L "$OUTPUT_ROOT/selections.jsonl" ]]
[[ -f "$OUTPUT_ROOT/report.json" && ! -L "$OUTPUT_ROOT/report.json" ]]
chmod -R a-w "$OUTPUT_ROOT"
