#!/bin/bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36-mtr-stack-%j.out
#SBATCH --error=logs/q36-mtr-stack-%j.err

set -euo pipefail

for name in PYTHON STACK_SCRIPT STACK_SCRIPT_SHA256 CROSS_VALIDATION_REPORT PRODUCTION_CANDIDATES PRODUCTION_SCORES CURRENT_CANDIDATES OWNER71_CANDIDATES OWNER8_CANDIDATES OUTPUT DECISIONS REPORT; do
  [[ -n "${!name:-}" ]] || { printf 'q36 sparse stack missing %s\n' "$name" >&2; exit 2; }
done
[[ -x "$PYTHON" && -f "$STACK_SCRIPT" && ! -L "$STACK_SCRIPT" ]] || { printf 'q36 sparse stack executable differs\n' >&2; exit 2; }
[[ "$(sha256sum "$STACK_SCRIPT" | awk '{print $1}')" == "$STACK_SCRIPT_SHA256" ]] || { printf 'q36 sparse stack script differs\n' >&2; exit 2; }
for path in "$OUTPUT" "$DECISIONS" "$REPORT"; do
  [[ ! -e "$path" && ! -L "$path" ]] || { printf 'q36 sparse stack output exists: %s\n' "$path" >&2; exit 2; }
done

args=(--cross-validation-report "$CROSS_VALIDATION_REPORT" --stacker "${STACKER:-hierarchical}" --output "$OUTPUT" --decisions "$DECISIONS" --report "$REPORT")
for specification in production:PRODUCTION current:CURRENT owner71:OWNER71 owner8:OWNER8; do
  owner=${specification%%:*}; prefix=${specification##*:}; candidates_name="${prefix}_CANDIDATES"
  IFS=: read -r -a candidates <<< "${!candidates_name}"
  [[ ${#candidates[@]} -eq 16 ]] || { printf 'q36 sparse stack candidate geometry differs\n' >&2; exit 2; }
  for path in "${candidates[@]}"; do args+=(--"$owner"-candidates "$path"); done
done
IFS=: read -r -a scores <<< "$PRODUCTION_SCORES"
[[ ${#scores[@]} -eq 16 ]] || { printf 'q36 sparse stack score geometry differs\n' >&2; exit 2; }
for path in "${scores[@]}"; do args+=(--production-scores "$path"); done

export PYTHONPATH="$(dirname "$STACK_SCRIPT")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PYTHON" -P -s -B "$STACK_SCRIPT" "${args[@]}"
chmod a-w "$OUTPUT" "$DECISIONS" "$REPORT"
