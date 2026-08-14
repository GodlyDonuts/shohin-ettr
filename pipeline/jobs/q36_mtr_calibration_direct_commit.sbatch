#!/bin/bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36-mtr-calibration-direct-%j.out
#SBATCH --error=logs/q36-mtr-calibration-direct-%j.err

set -euo pipefail

for name in PYTHON RUNTIME COMMIT_SCRIPT COMMIT_SCRIPT_SHA256 TRAINING_ROWS DEVELOPMENT_ROWS CURRENT_CANDIDATES OWNER71_CANDIDATES OWNER8_CANDIDATES MODEL_OUTPUT OUTPUT DECISIONS REPORT; do
  [[ -n "${!name:-}" ]] || { printf 'q36 calibration direct commit missing %s\n' "$name" >&2; exit 2; }
done
[[ -x "$PYTHON" && -f "$COMMIT_SCRIPT" && ! -L "$COMMIT_SCRIPT" ]] || { printf 'q36 calibration direct executable differs\n' >&2; exit 2; }
[[ "$(sha256sum "$COMMIT_SCRIPT" | awk '{print $1}')" == "$COMMIT_SCRIPT_SHA256" ]] || { printf 'q36 calibration direct script differs\n' >&2; exit 2; }
for path in "$MODEL_OUTPUT" "$OUTPUT" "$DECISIONS" "$REPORT"; do [[ ! -e "$path" && ! -L "$path" ]] || { printf 'q36 calibration direct output exists: %s\n' "$path" >&2; exit 2; }; done

args=(--training-rows "$TRAINING_ROWS" --development-rows "$DEVELOPMENT_ROWS" --model-output "$MODEL_OUTPUT" --output "$OUTPUT" --decisions "$DECISIONS" --report "$REPORT")
for owner in current owner71 owner8; do
  name="${owner^^}_CANDIDATES"; IFS=: read -r -a candidates <<< "${!name}"
  [[ ${#candidates[@]} -eq 16 ]] || { printf 'q36 calibration direct shard geometry differs\n' >&2; exit 2; }
  for path in "${candidates[@]}"; do args+=(--"$owner"-candidates "$path"); done
done
export PYTHONPATH="$RUNTIME/pipeline:$RUNTIME/train"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PYTHON" -P -s -B "$COMMIT_SCRIPT" "${args[@]}"
chmod a-w "$MODEL_OUTPUT" "$OUTPUT" "$DECISIONS" "$REPORT"
