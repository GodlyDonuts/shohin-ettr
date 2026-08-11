#!/bin/bash
#SBATCH --job-name=pcf1-normalize
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc38,evc46
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/pcf1_normalize_%j.out
#SBATCH --error=logs/pcf1_normalize_%j.err

set -euo pipefail

: "${RUNTIME:?}" "${RUNTIME_MANIFEST_SHA256:?}"
test -f "$RUNTIME/SHA256SUMS"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
source "$RUNTIME/train/jobs/pcf1_common.sh"
pcf1_verify_runtime
for variable in LEARNED_COMMIT_REPORT REVISION_REPORT UNCHANGED_REPORT SELF_REFINEMENT_REPORT DATA_CUSTODY MODEL_CUSTODY RUNTIME_CUSTODY SCORE_CONSUMPTION; do
  pcf1_require "$variable"
  pcf1_require_safe_input "${!variable}"
done
pcf1_require OUTPUT_ROOT
pcf1_require_fresh "$OUTPUT_ROOT"
python=$(pcf1_python)
pcf1_export_runtime
"$python" "$RUNTIME/pipeline/normalize_pcf1_reports.py" \
  --learned-commit-report "$LEARNED_COMMIT_REPORT" \
  --revision-report "$REVISION_REPORT" \
  --unchanged-report "$UNCHANGED_REPORT" \
  --self-refinement-report "$SELF_REFINEMENT_REPORT" \
  --data-custody "$DATA_CUSTODY" \
  --model-custody "$MODEL_CUSTODY" \
  --runtime-custody "$RUNTIME_CUSTODY" \
  --score-consumption "$SCORE_CONSUMPTION" \
  --output "$OUTPUT_ROOT"
for arm in learned_commit trained_revision unchanged self_refinement; do
  pcf1_validate_json "$OUTPUT_ROOT/$arm.json" shohin-pcf1-arm-report-v1
done
pcf1_freeze_tree "$OUTPUT_ROOT"
