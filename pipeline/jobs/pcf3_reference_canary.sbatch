#!/bin/bash
# Non-scientific trusted-reference admission for the PCF3 successor.
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc38,evc46
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail

source "${RUNTIME:?}/train/jobs/pcf1_common.sh"
pcf1_require PYTHON
pcf1_require SOURCE_BANK
pcf1_require OUTPUT_ROOT
pcf1_require_file "$SOURCE_BANK"
[[ "$(pcf1_sha256 "$SOURCE_BANK")" == \
  0b6d068b4d71f407cb234579b9278dc640df09139ea906dd0f52a6ab71e05398 ]] || \
  pcf1_die "PCF3 frozen MBPP source differs"
pcf1_require_fresh "$OUTPUT_ROOT"
pcf1_initialize_scratch
freeze_reference_canary() {
  local status=$?
  trap - EXIT
  [[ ! -e "$OUTPUT_ROOT" ]] || chmod -R a-w "$OUTPUT_ROOT"
  if ! pcf1_cleanup_scratch; then
    [[ "$status" -ne 0 ]] || status=1
  fi
  exit "$status"
}
trap freeze_reference_canary EXIT
pcf1_export_runtime
"$PYTHON" "$RUNTIME/pipeline/qualify_pcf3_references.py" \
  --source-bank "$SOURCE_BANK" \
  --output-root "$OUTPUT_ROOT"
pcf1_validate_json "$OUTPUT_ROOT/report.json" shohin-pcf3-reference-canary-v1 pass
pcf1_freeze_tree "$OUTPUT_ROOT"
pcf1_cleanup_scratch
trap - EXIT
