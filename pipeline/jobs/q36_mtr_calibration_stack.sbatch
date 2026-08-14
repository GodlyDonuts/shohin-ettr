#!/bin/bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36-mtr-calibration-stack-%j.out
#SBATCH --error=logs/q36-mtr-calibration-stack-%j.err

set -euo pipefail

for name in PYTHON STACK_SCRIPT STACK_SCRIPT_SHA256 TRAINING_ROWS DEVELOPMENT_ROWS CURRENT_CANDIDATES OWNER71_CANDIDATES OWNER8_CANDIDATES MODEL_OUTPUT OUTPUT DECISIONS REPORT; do
  [[ -n "${!name:-}" ]] || { printf 'q36 calibration stack missing %s\n' "$name" >&2; exit 2; }
done
[[ -x "$PYTHON" && -f "$STACK_SCRIPT" && ! -L "$STACK_SCRIPT" ]] || { printf 'q36 calibration stack executable differs\n' >&2; exit 2; }
[[ "$(sha256sum "$STACK_SCRIPT" | awk '{print $1}')" == "$STACK_SCRIPT_SHA256" ]] || { printf 'q36 calibration stack script differs\n' >&2; exit 2; }
for path in "$MODEL_OUTPUT" "$OUTPUT" "$DECISIONS" "$REPORT"; do [[ ! -e "$path" && ! -L "$path" ]] || { printf 'q36 calibration stack output exists: %s\n' "$path" >&2; exit 2; }; done

args=(--training-rows "$TRAINING_ROWS" --development-rows "$DEVELOPMENT_ROWS" --model-output "$MODEL_OUTPUT" --output "$OUTPUT" --decisions "$DECISIONS" --report "$REPORT")
for owner in current owner71 owner8; do
  name="${owner^^}_CANDIDATES"; IFS=: read -r -a candidates <<< "${!name}"
  [[ ${#candidates[@]} -eq 16 ]] || { printf 'q36 calibration stack shard geometry differs\n' >&2; exit 2; }
  for path in "${candidates[@]}"; do args+=(--"$owner"-candidates "$path"); done
done
export PYTHONPATH="$(dirname "$STACK_SCRIPT")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PYTHON" -P -s -B "$STACK_SCRIPT" "${args[@]}"
chmod a-w "$MODEL_OUTPUT" "$OUTPUT" "$DECISIONS" "$REPORT"
