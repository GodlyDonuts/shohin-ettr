#!/bin/bash
#SBATCH --job-name=nemotron-super-ultra-lift
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail

required=(
  RUNTIME PYTHON SUPER_CHECKPOINT SUPER_CHECKPOINT_SHA256 TRANSFER_FACTOR
  ULTRA_CONFIG OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]] || {
  printf 'output root exists\n' >&2
  exit 2
}
mkdir -p "$OUTPUT_ROOT"
trap 'status=$?; if (( status != 0 )); then chmod -R a-w "$OUTPUT_ROOT" 2>/dev/null || true; fi' EXIT
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
"$PYTHON" -P -s -B "$RUNTIME/pipeline/lift_nemotron_super_adapter_to_ultra.py" \
  --super-checkpoint "$SUPER_CHECKPOINT" \
  --expected-super-checkpoint-sha256 "$SUPER_CHECKPOINT_SHA256" \
  --factor "$TRANSFER_FACTOR" \
  --ultra-config "$ULTRA_CONFIG" \
  --output-checkpoint "$OUTPUT_ROOT/checkpoint.pt" \
  --output-report "$OUTPUT_ROOT/report.json"
test -f "$OUTPUT_ROOT/checkpoint.pt" && test -f "$OUTPUT_ROOT/report.json"
chmod -R a-w "$OUTPUT_ROOT"
