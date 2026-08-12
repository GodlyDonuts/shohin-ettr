#!/bin/bash
#SBATCH --job-name=pcf1-final-gate
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc33,evc37,evc38,evc46
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/pcf1_final_gate_%j.out
#SBATCH --error=logs/pcf1_final_gate_%j.err

set -euo pipefail

: "${RUNTIME:?}" "${RUNTIME_MANIFEST_SHA256:?}"
test -f "$RUNTIME/SHA256SUMS"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
source "$RUNTIME/train/jobs/pcf1_common.sh"
pcf1_verify_runtime
for variable in LEARNED_COMMIT_REPORT TRAINED_REVISION_REPORT UNCHANGED_REPORT SELF_REFINEMENT_REPORT DATA_CUSTODY MODEL_CUSTODY RUNTIME_CUSTODY COMPUTE_CUSTODY OUTPUT; do
  pcf1_require "$variable"
done
for path in "$LEARNED_COMMIT_REPORT" "$TRAINED_REVISION_REPORT" "$UNCHANGED_REPORT" "$SELF_REFINEMENT_REPORT" "$DATA_CUSTODY" "$MODEL_CUSTODY" "$RUNTIME_CUSTODY" "$COMPUTE_CUSTODY"; do
  pcf1_require_safe_input "$path"
done
pcf1_require_fresh "$OUTPUT"
python=$(pcf1_python)
pcf1_export_runtime
set +e
"$python" "$RUNTIME/pipeline/compare_pcf1.py" \
  --learned-commit-report "$LEARNED_COMMIT_REPORT" \
  --trained-revision-report "$TRAINED_REVISION_REPORT" \
  --unchanged-report "$UNCHANGED_REPORT" \
  --self-refinement-report "$SELF_REFINEMENT_REPORT" \
  --data-custody "$DATA_CUSTODY" \
  --model-custody "$MODEL_CUSTODY" \
  --runtime-custody "$RUNTIME_CUSTODY" \
  --compute-custody "$COMPUTE_CUSTODY" \
  --output "$OUTPUT"
status=$?
set -e
pcf1_validate_json "$OUTPUT" shohin-pcf1-final-comparison-v1
chmod a-w "$OUTPUT"
case "$status" in
  0|3) exit "$status" ;;
  *) pcf1_die "PCF1 final comparator failed before sealing a gate result" ;;
esac
