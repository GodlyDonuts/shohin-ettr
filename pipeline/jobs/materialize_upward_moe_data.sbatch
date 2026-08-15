#!/bin/bash
#SBATCH --job-name=materialize-upward-moe
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=logs/materialize_upward_moe_%j.out
#SBATCH --error=logs/materialize_upward_moe_%j.err

set -euo pipefail
for variable in HOST PYTHON RUNTIME SOURCE_ROOT DRAFTS DRAFT_REPORT ASSESSOR_RECEIPT OUTPUT; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
[[ "$HOST" == nemotron-super || "$HOST" == mixtral-8x22b || \
  "$HOST" == nemotron-ultra ]]
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]]
mkdir -p "$(dirname "$OUTPUT")" logs
PYTHONPATH="$RUNTIME/pipeline:$RUNTIME/train" "$PYTHON" -P -s -B \
  "$RUNTIME/pipeline/build_q36_mtr_data.py" \
  --source-root "$SOURCE_ROOT" --drafts "$DRAFTS" \
  --draft-report "$DRAFT_REPORT" --assessor-receipt "$ASSESSOR_RECEIPT" \
  --upward-host "$HOST" --output "$OUTPUT"
chmod -R a-w "$OUTPUT"
