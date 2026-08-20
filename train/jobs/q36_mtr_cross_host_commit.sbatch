#!/bin/bash
#SBATCH --job-name=q36-cross-host-commit
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc33,evc34,evc37,evc38,evc43,evc46,evc50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_cross_host_commit_%j.out
#SBATCH --error=logs/q36_cross_host_commit_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE_COMMIT PYTHON MODEL_ROOT ALIGNED_CHECKPOINT COMMIT_CHECKPOINT HOST SOURCE REVISION_CANDIDATES UNCHANGED_CANDIDATES OUTPUT SELECTIONS REPORT ENVIRONMENT_RECEIPT ENVIRONMENT_RECEIPT_SHA256 ENVIRONMENT_TREE_SHA256 CROSS_HOST_CONTRACT CROSS_HOST_CONTRACT_SHA256 RUN_ROOT; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
q36_verify_runtime
q36_verify_model
q36_verify_environment
q36_verify_sha256 "$CROSS_HOST_CONTRACT" "$CROSS_HOST_CONTRACT_SHA256"
"$PYTHON" - "$CROSS_HOST_CONTRACT" "$HOST" "$RUN_ROOT" "$OUTPUT" "$SELECTIONS" "$REPORT" "${REVISION_MARGIN_THRESHOLD:-0.0}" "${REVISION_RELIABILITY_VETO:-none}" <<'PY'
import json
from pathlib import Path
import sys

contract = json.load(open(sys.argv[1], encoding="utf-8"))
root = Path(sys.argv[3]).resolve(strict=False)
threshold = float(sys.argv[7])
reliability_veto = sys.argv[8]
if (
    contract.get("schema") != "shohin-q36-cross-host-model-owned-commit-contract-v1"
    or contract.get("status") != "frozen_before_execution"
    or sys.argv[2] not in contract.get("hosts", {})
    or root in {Path("/"), Path.home().resolve()}
    or threshold
    != float(contract.get("selector", {}).get("revision_margin_threshold", 0.0))
    or reliability_veto
    != contract.get("selector", {}).get("revision_reliability_veto", "none")
):
    raise SystemExit("cross-host authorization differs")
for value in sys.argv[4:7]:
    target = Path(value).resolve(strict=False)
    try:
        relative = target.relative_to(root)
    except ValueError as error:
        raise SystemExit("cross-host output escapes run root") from error
    if not relative.parts:
        raise SystemExit("cross-host output equals run root")
PY
case "$HOST" in
  gpt_oss_120b_screen|gpt_oss_120b_confirmation) expected_shards=4 ;;
  mixtral_8x22b_validation) expected_shards=16 ;;
  *) q36_die "cross-host identity differs" ;;
esac
IFS=: read -r -a revision_paths <<< "$REVISION_CANDIDATES"
IFS=: read -r -a unchanged_paths <<< "$UNCHANGED_CANDIDATES"
[[ ${#revision_paths[@]} -eq $expected_shards && ${#unchanged_paths[@]} -eq $expected_shards ]] || q36_die "cross-host shard geometry differs"
for path in "$ALIGNED_CHECKPOINT" "$COMMIT_CHECKPOINT" "$SOURCE" "${revision_paths[@]}" "${unchanged_paths[@]}"; do
  [[ -f "$path" && ! -L "$path" ]] || q36_die "cross-host input differs"
done
for path in "$OUTPUT" "$SELECTIONS" "$REPORT"; do
  [[ ! -e "$path" && ! -L "$path" ]] || q36_die "cross-host output exists"
done
model=$(q36_stage_model)
mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$SELECTIONS")" "$(dirname "$REPORT")" logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
q36_export_pythonpath
arguments=()
for path in "${revision_paths[@]}"; do arguments+=(--revision-candidates "$path"); done
for path in "${unchanged_paths[@]}"; do arguments+=(--unchanged-candidates "$path"); done
nvidia-smi
"$PYTHON" -P -s -B "$RUNTIME/train/hf_q36_mtr_apply_cross_host_commit.py" \
  --host "$HOST" \
  --model-root "$model" \
  --model-revision "$MODEL_REVISION" \
  --adapter-checkpoint "$ALIGNED_CHECKPOINT" \
  --commit-checkpoint "$COMMIT_CHECKPOINT" \
  --source "$SOURCE" \
  "${arguments[@]}" \
  --output "$OUTPUT" \
  --selections "$SELECTIONS" \
  --report "$REPORT" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --environment-receipt-sha256 "$ENVIRONMENT_RECEIPT_SHA256" \
  --environment-tree-sha256 "$ENVIRONMENT_TREE_SHA256" \
  --max-sequence-length 3072 --batch-identities 2 \
  --revision-margin-threshold "${REVISION_MARGIN_THRESHOLD:-0.0}" \
  --revision-reliability-veto "${REVISION_RELIABILITY_VETO:-none}"
test -f "$OUTPUT" && test -f "$SELECTIONS" && test -f "$REPORT"
chmod a-w "$OUTPUT" "$SELECTIONS" "$REPORT"
