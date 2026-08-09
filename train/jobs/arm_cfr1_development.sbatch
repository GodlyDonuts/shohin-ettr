#!/bin/bash
#SBATCH --job-name=cfr1-eval-arm
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/cfr1_eval_arm_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/cfr1_eval_arm_%j.err

set -euo pipefail

for variable in RUNTIME RUNTIME_MANIFEST_SHA256 FIT_DISPATCH_LOG MODEL_ROOT \
  MODEL_REVISION MODEL_CONFIG_SHA256 MODEL_MANIFEST_SHA256 DEVELOPMENT \
  DEVELOPMENT_REPORT CFR_DATA_REPORT ALIGNED_TRAIN_ROOT SHUFFLED_TRAIN_ROOT \
  OUTPUT_ROOT RECEIPT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done

test -f "$FIT_DISPATCH_LOG"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = \
  "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test ! -e "$OUTPUT_ROOT"
test ! -e "$RECEIPT"

extract_one_job() {
  local key=$1
  local values
  values=$(sed -n "s/^${key}=\([0-9][0-9]*\)$/\1/p" "$FIT_DISPATCH_LOG")
  test "$(printf '%s\n' "$values" | sed '/^$/d' | wc -l)" -eq 1
  printf '%s' "$values"
}

aligned_job=$(extract_one_job ALIGNED)
shuffled_job=$(extract_one_job SHUFFLED)
test "$aligned_job" != "$shuffled_job"

development_sha256=$(sha256sum "$DEVELOPMENT" | cut -d' ' -f1)
development_report_sha256=$(sha256sum "$DEVELOPMENT_REPORT" | cut -d' ' -f1)
cfr_data_report_sha256=$(sha256sum "$CFR_DATA_REPORT" | cut -d' ' -f1)

exports="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
exports+=",MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION"
exports+=",MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256"
exports+=",MODEL_MANIFEST_SHA256=$MODEL_MANIFEST_SHA256"
exports+=",DEVELOPMENT=$DEVELOPMENT,DEVELOPMENT_SHA256=$development_sha256"
exports+=",DEVELOPMENT_REPORT=$DEVELOPMENT_REPORT"
exports+=",DEVELOPMENT_REPORT_SHA256=$development_report_sha256"
exports+=",CFR_DATA_REPORT=$CFR_DATA_REPORT"
exports+=",CFR_DATA_REPORT_SHA256=$cfr_data_report_sha256"
exports+=",ALIGNED_TRAIN_ROOT=$ALIGNED_TRAIN_ROOT"
exports+=",SHUFFLED_TRAIN_ROOT=$SHUFFLED_TRAIN_ROOT"
exports+=",OUTPUT_ROOT=$OUTPUT_ROOT,RECEIPT=$RECEIPT"

dispatcher=$(sbatch --parsable \
  --dependency=afterok:"$aligned_job":"$shuffled_job" \
  --export="$exports" "$RUNTIME/train/jobs/dispatch_cfr1_development.sbatch")
printf 'EVALUATION_DISPATCH=%s\nALIGNED=%s\nSHUFFLED=%s\n' \
  "$dispatcher" "$aligned_job" "$shuffled_job"
