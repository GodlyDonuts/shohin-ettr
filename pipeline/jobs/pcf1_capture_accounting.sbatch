#!/bin/bash
#SBATCH --job-name=pcf1-accounting
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc33,evc37,evc38,evc46
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/pcf1_accounting_%j.out
#SBATCH --error=logs/pcf1_accounting_%j.err

set -euo pipefail

: "${RUNTIME:?}" "${RUNTIME_MANIFEST_SHA256:?}"
test -f "$RUNTIME/SHA256SUMS"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
source "$RUNTIME/train/jobs/pcf1_common.sh"
pcf1_verify_runtime
for variable in RUN_ID DISPATCH_RECEIPT OUTPUT; do
  pcf1_require "$variable"
done
pcf1_require_safe_input "$DISPATCH_RECEIPT"
pcf1_require_fresh "$OUTPUT"
python=$(pcf1_python)
pcf1_export_runtime
"$python" "$RUNTIME/pipeline/capture_pcf1_slurm_accounting.py" \
  --run-id "$RUN_ID" \
  --dispatch-receipt "$DISPATCH_RECEIPT" \
  --output "$OUTPUT"
pcf1_validate_json "$OUTPUT" shohin-pcf1-slurm-accounting-v1
chmod a-w "$OUTPUT"
