#!/bin/bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36-mtr-calibration-correctness-%j.out
#SBATCH --error=logs/q36-mtr-calibration-correctness-%j.err

set -euo pipefail

for name in PYTHON RUNTIME CORRECTNESS_SCRIPT CORRECTNESS_SCRIPT_SHA256 TRAINING_ROWS DEVELOPMENT_ROWS MODEL_OUTPUT OUTPUT PRODUCTION_OUTPUT DECISIONS REPORT; do
  [[ -n "${!name:-}" ]] || { printf 'q36 calibration correctness missing %s\n' "$name" >&2; exit 2; }
done
if [[ "${EMBEDDED_DEVELOPMENT_CANDIDATES:-0}" != 1 ]]; then
  for name in CURRENT_CANDIDATES OWNER71_CANDIDATES OWNER8_CANDIDATES; do
    [[ -n "${!name:-}" ]] || { printf 'q36 calibration correctness missing %s\n' "$name" >&2; exit 2; }
  done
fi
[[ -x "$PYTHON" && -f "$CORRECTNESS_SCRIPT" && ! -L "$CORRECTNESS_SCRIPT" ]] || exit 2
[[ "$(sha256sum "$CORRECTNESS_SCRIPT" | awk '{print $1}')" == "$CORRECTNESS_SCRIPT_SHA256" ]] || exit 2
for path in "$MODEL_OUTPUT" "$OUTPUT" "$PRODUCTION_OUTPUT" "$DECISIONS" "$REPORT"; do [[ ! -e "$path" && ! -L "$path" ]] || exit 2; done
args=(--training-rows "$TRAINING_ROWS" --development-rows "$DEVELOPMENT_ROWS" --model-output "$MODEL_OUTPUT" --output "$OUTPUT" --production-output "$PRODUCTION_OUTPUT" --decisions "$DECISIONS" --report "$REPORT")
if [[ "${EMBEDDED_DEVELOPMENT_CANDIDATES:-0}" == 1 ]]; then
  args+=(--embedded-development-candidates)
else
  for owner in current owner71 owner8; do
    name="${owner^^}_CANDIDATES"; IFS=: read -r -a candidates <<< "${!name}"
    [[ ${#candidates[@]} -eq 16 ]] || exit 2
    for path in "${candidates[@]}"; do args+=(--"$owner"-candidates "$path"); done
  done
fi
if [[ -n "${REUSE_DEVELOPMENT_DECISIONS:-}" ]]; then
  [[ -f "$REUSE_DEVELOPMENT_DECISIONS" && ! -L "$REUSE_DEVELOPMENT_DECISIONS" ]] || exit 2
  args+=(--reuse-development-decisions "$REUSE_DEVELOPMENT_DECISIONS")
fi
export PYTHONPATH="$RUNTIME/pipeline:$RUNTIME/train"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PYTHON" -P -s -B "$CORRECTNESS_SCRIPT" "${args[@]}"
chmod a-w "$MODEL_OUTPUT" "$OUTPUT" "$PRODUCTION_OUTPUT" "$DECISIONS" "$REPORT"
