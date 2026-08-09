#!/bin/bash
#SBATCH --job-name=ndr1-fit-dispatch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/ndr1_fit_dispatch_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/ndr1_fit_dispatch_%j.err

set -euo pipefail

for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION \
  MODEL_CONFIG_SHA256 WARM_START WARM_START_SHA256 DATA_ROOT OUTPUT_ROOT \
  RECEIPT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
EXCLUDE_NODES=${EXCLUDE_NODES:-evc26:evc31:evc33:evc38:evc46}
EXCLUDE=${EXCLUDE_NODES//:/,}
test ! -e "$OUTPUT_ROOT"
test ! -e "$RECEIPT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = \
  "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)

report=$DATA_ROOT/report.json
aligned=$DATA_ROOT/train_aligned.jsonl
shuffled=$DATA_ROOT/train_shuffled.jsonl
report_sha=$(sha256sum "$report" | cut -d' ' -f1)
aligned_sha=$(sha256sum "$aligned" | cut -d' ' -f1)
shuffled_sha=$(sha256sum "$shuffled" | cut -d' ' -f1)
mkdir -p "$OUTPUT_ROOT" "$(dirname "$RECEIPT")"

common="ALL,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
common+=",MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION"
common+=",MODEL_CONFIG_SHA256=$MODEL_CONFIG_SHA256,MODEL_LOADER=multimodal"
common+=",WARM_START=$WARM_START,WARM_START_SHA256=$WARM_START_SHA256"
common+=",DATA_REPORT=$report,DATA_REPORT_SHA256=$report_sha"

mechanics=$(sbatch --parsable --exclude="$EXCLUDE" \
  --export="$common,DATA=$aligned,DATA_SHA256=$aligned_sha,OUTPUT=$OUTPUT_ROOT/mechanics_u1,ARM=aligned,UPDATES=1,CHECKPOINT_INTERVAL=1" \
  "$RUNTIME/train/jobs/hf_ndr1_train_reviser.sbatch")
aligned_job=$(sbatch --parsable --exclude="$EXCLUDE" --dependency=afterok:"$mechanics" \
  --export="$common,DATA=$aligned,DATA_SHA256=$aligned_sha,OUTPUT=$OUTPUT_ROOT/train_aligned_u512,ARM=aligned" \
  "$RUNTIME/train/jobs/hf_ndr1_train_reviser.sbatch")
shuffled_job=$(sbatch --parsable --exclude="$EXCLUDE" --dependency=afterok:"$mechanics" \
  --export="$common,DATA=$shuffled,DATA_SHA256=$shuffled_sha,OUTPUT=$OUTPUT_ROOT/train_shuffled_u512,ARM=shuffled" \
  "$RUNTIME/train/jobs/hf_ndr1_train_reviser.sbatch")

temporary=$RECEIPT.partial.$$
printf '{"aligned_job":"%s","mechanics_job":"%s","shuffled_job":"%s","status":"submitted"}\n' \
  "$aligned_job" "$mechanics" "$shuffled_job" >"$temporary"
mv "$temporary" "$RECEIPT"
chmod a-w "$RECEIPT"
printf 'MECHANICS=%s\nALIGNED=%s\nSHUFFLED=%s\n' \
  "$mechanics" "$aligned_job" "$shuffled_job"
