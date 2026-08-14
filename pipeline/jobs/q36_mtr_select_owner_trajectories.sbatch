#!/bin/bash
#SBATCH --job-name=q36-mtr-owner-select
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_mtr_owner_select_%j.out
#SBATCH --error=logs/q36_mtr_owner_select_%j.err

set -euo pipefail

for variable in PYTHON RUNTIME SELECTOR_SCRIPT SELECTOR_SCRIPT_SHA256 FIRST_CANDIDATES SECOND_CANDIDATES OUTPUT REPORT; do
  [[ -n "$(printenv "$variable")" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
[[ "$SELECTOR_SCRIPT_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
[[ "$(sha256sum "$SELECTOR_SCRIPT" | cut -d' ' -f1)" == "$SELECTOR_SCRIPT_SHA256" ]] || exit 2
[[ -f "$FIRST_CANDIDATES" && ! -L "$FIRST_CANDIDATES" ]] || exit 2
[[ -f "$SECOND_CANDIDATES" && ! -L "$SECOND_CANDIDATES" ]] || exit 2
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" && ! -e "$REPORT" && ! -L "$REPORT" ]] || exit 2
export PATH=/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export PYTHONPATH=$RUNTIME/train:$RUNTIME/pipeline
"$PYTHON" "$SELECTOR_SCRIPT" \
  --first-candidates "$FIRST_CANDIDATES" \
  --second-candidates "$SECOND_CANDIDATES" \
  --output "$OUTPUT" \
  --report "$REPORT"
test -f "$OUTPUT" && test -f "$REPORT"
chmod a-w "$OUTPUT" "$REPORT"
