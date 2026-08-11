#!/bin/bash
#SBATCH --job-name=pcf1-prepare
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc33,evc38,evc46
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pcf1_prepare_%j.out
#SBATCH --error=logs/pcf1_prepare_%j.err

set -euo pipefail

for variable in RUNTIME RUNTIME_MANIFEST_SHA256 PYTHON MODEL_ROOT MODEL_REVISION PAIRS MATH_BANK SCIENCE_BANK CODE_BANK B1_DATA_LEGACY OUTPUT ASSESSOR_OUTPUT ASSESSOR_RECEIPT_OUTPUT CPU_RECEIPT_OUTPUT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
test -x "$PYTHON"
test -f "$RUNTIME/SHA256SUMS"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
source "$RUNTIME/train/jobs/pcf1_common.sh"
pcf1_verify_runtime_membership
for output_path in "$OUTPUT" "$ASSESSOR_OUTPUT" "$ASSESSOR_RECEIPT_OUTPUT" "$CPU_RECEIPT_OUTPUT"; do
  pcf1_require_fresh "$output_path"
done
test "$MODEL_REVISION" = 81eaece1948f3875421d9a45bc55487d10e2d894
test "$(realpath "$MODEL_ROOT")" = /lustre/fs1/home/sa305415/shohin/artifacts/external/ministral-3-8b-reasoning-2512-81eaece
test ! -e "$OUTPUT"
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
pcf1_export_runtime
environment_receipt=$SLURM_TMPDIR/pcf1-environment-receipt.json
"$PYTHON" "$RUNTIME/pipeline/capture_pcf1_environment.py" capture \
  --runtime-root "$RUNTIME" \
  --runtime-manifest-sha256 "$RUNTIME_MANIFEST_SHA256" \
  --output "$environment_receipt"
"$PYTHON" "$RUNTIME/pipeline/prepare_pcf1_inputs.py" \
  --pairs "$PAIRS" \
  --math-bank "$MATH_BANK" \
  --science-bank "$SCIENCE_BANK" \
  --code-bank "$CODE_BANK" \
  --b1-data "$B1_DATA_LEGACY" \
  --model-root "$MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --environment-receipt "$environment_receipt" \
  --assessor-output "$ASSESSOR_OUTPUT" \
  --assessor-receipt-output "$ASSESSOR_RECEIPT_OUTPUT" \
  --cpu-receipt-output "$CPU_RECEIPT_OUTPUT" \
  --output "$OUTPUT"
"$PYTHON" - "$OUTPUT/receipt.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("schema") != "shohin-pcf1-prepare-receipt-v1" or report.get("status") != "complete":
    raise SystemExit("PCF1 prepare receipt differs")
PY
chmod -R a-w "$OUTPUT"
chmod a-w "$ASSESSOR_OUTPUT" "$ASSESSOR_RECEIPT_OUTPUT" "$CPU_RECEIPT_OUTPUT"
