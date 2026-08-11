#!/bin/bash
#SBATCH --job-name=pcf1-eval
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc38,evc46
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/pcf1_eval_%A_%a.out
#SBATCH --error=logs/pcf1_eval_%A_%a.err

set -euo pipefail

: "${RUNTIME:?}" "${RUNTIME_MANIFEST_SHA256:?}"
test -f "$RUNTIME/SHA256SUMS"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
source "$RUNTIME/train/jobs/pcf1_common.sh"
pcf1_assert_gpu_environment
pcf1_verify_runtime
pcf1_bind_prepared_inputs
pcf1_verify_model
for variable in ARM SPLIT DATA DATA_REPORT ADAPTER_CHECKPOINT OUTPUT_ROOT EVAL_SHARDS; do
  pcf1_require "$variable"
done
case "$SPLIT:$ARM" in
  calibration:revision|calibration:unchanged|confirmation:revision|confirmation:unchanged|confirmation:self_refinement) ;;
  *) pcf1_die "unauthorized PCF1 evaluation arm/split: $SPLIT:$ARM" ;;
esac
[[ "$EVAL_SHARDS" == 4 ]] || pcf1_die "EVAL_SHARDS must equal frozen value 4"
[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || pcf1_die "evaluation must run as an array"
((SLURM_ARRAY_TASK_ID < EVAL_SHARDS)) || pcf1_die "evaluation shard index differs"
pcf1_require_safe_input "$DATA"
pcf1_require_safe_input "$DATA_REPORT"
pcf1_require_safe_input "$ADAPTER_CHECKPOINT"
pcf1_validate_json "$DATA_REPORT" shohin-pcf1-data-report-v1
shard_root=$OUTPUT_ROOT/shards/$SLURM_ARRAY_TASK_ID
pcf1_require_fresh "$shard_root/candidates.jsonl"
pcf1_require_fresh "$shard_root/report.json"
sandbox_probe=
pcf1_initialize_scratch
freeze_sandbox_probe() {
  local status=$?
  trap - EXIT
  [[ -z "$sandbox_probe" || ! -f "$sandbox_probe" ]] || chmod a-w "$sandbox_probe"
  if ! pcf1_cleanup_scratch; then
    [[ "$status" -ne 0 ]] || status=1
  fi
  exit "$status"
}
trap freeze_sandbox_probe EXIT
extra_arguments=()
if [[ "$SPLIT" == calibration ]]; then
  pcf1_require MECHANICS_REPORT
  pcf1_require SANDBOX_RECEIPT
  pcf1_require_safe_input "$MECHANICS_REPORT"
  pcf1_validate_json "$MECHANICS_REPORT" shohin-pcf1-mechanics-v1 pass
  sandbox_receipt_sha256=$(pcf1_validate_sandbox_receipt "$SANDBOX_RECEIPT")
  python=$(pcf1_python)
  "$python" - "$MECHANICS_REPORT" "$SANDBOX_RECEIPT" "$sandbox_receipt_sha256" <<'PY'
import json
from pathlib import Path
import sys

mechanics_path, sandbox_path = map(Path, sys.argv[1:3])
expected_sha256 = sys.argv[3]
mechanics = json.loads(mechanics_path.read_text(encoding="utf-8"))
if (
    Path(str(mechanics.get("sandbox_receipt", ""))).resolve()
    != sandbox_path.resolve()
    or mechanics.get("sandbox_receipt_sha256") != expected_sha256
    or mechanics.get("sandbox_isolation_passed") is not True
):
    raise SystemExit("PCF1 mechanics sandbox binding differs")
PY
  sandbox_probe=$shard_root/sandbox_probe.json
  pcf1_require_fresh "$sandbox_probe"
  extra_arguments+=(--sandbox-probe-output "$sandbox_probe")
elif [[ -n "${MECHANICS_REPORT:-}" || -n "${SANDBOX_RECEIPT:-}" ]]; then
  pcf1_die "confirmation generation must not receive sandbox admission paths"
fi
python=$(pcf1_python)
pcf1_initialize_scratch
model=$(pcf1_stage_model)
pcf1_export_runtime
nvidia-smi
"$python" "$RUNTIME/train/hf_pcf1_evaluate.py" \
  --arm "$ARM" \
  --model-root "$model" \
  --model-source-root "$MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --model-loader multimodal \
  --adapter-checkpoint "$ADAPTER_CHECKPOINT" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --environment-receipt-sha256 "$ENVIRONMENT_RECEIPT_SHA256" \
  --data "$DATA" \
  --data-report "$DATA_REPORT" \
  --split "$SPLIT" \
  --candidates-output "$shard_root/candidates.jsonl" \
  --report "$shard_root/report.json" \
  "${extra_arguments[@]}" \
  --batch-size 2 \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --shard-count "$EVAL_SHARDS" \
  --seed 2026080816
pcf1_validate_json "$shard_root/report.json" shohin-pcf1-evaluation-v1
pcf1_freeze_tree "$shard_root"
pcf1_cleanup_scratch
trap - EXIT
