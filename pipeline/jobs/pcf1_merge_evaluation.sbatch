#!/bin/bash
#SBATCH --job-name=pcf1-eval-merge
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc38,evc46
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/pcf1_eval_merge_%j.out
#SBATCH --error=logs/pcf1_eval_merge_%j.err

set -euo pipefail

: "${RUNTIME:?}" "${RUNTIME_MANIFEST_SHA256:?}"
test -f "$RUNTIME/SHA256SUMS"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
source "$RUNTIME/train/jobs/pcf1_common.sh"
pcf1_verify_runtime
for variable in ARM SPLIT DATA DATA_REPORT SHARD_ROOT EVAL_SHARDS CANDIDATES_OUTPUT REPORT; do
  pcf1_require "$variable"
done
case "$SPLIT:$ARM" in
  calibration:revision|calibration:unchanged|confirmation:revision|confirmation:unchanged|confirmation:self_refinement) ;;
  *) pcf1_die "unauthorized PCF1 evaluation merge: $SPLIT:$ARM" ;;
esac
[[ "$EVAL_SHARDS" == 4 ]] || pcf1_die "EVAL_SHARDS must equal frozen value 4"
pcf1_require_safe_input "$DATA"
pcf1_require_safe_input "$DATA_REPORT"
pcf1_require_fresh "$CANDIDATES_OUTPUT"
pcf1_require_fresh "$REPORT"
arguments=()
for ((index = 0; index < EVAL_SHARDS; index++)); do
  shard_report=$SHARD_ROOT/shards/$index/report.json
  shard_candidates=$SHARD_ROOT/shards/$index/candidates.jsonl
  shard_sandbox_probe=$SHARD_ROOT/shards/$index/sandbox_probe.json
  pcf1_require_safe_input "$shard_report"
  pcf1_require_safe_input "$shard_candidates"
  arguments+=(--shard-report "$shard_report" --shard-candidates "$shard_candidates")
  if [[ "$SPLIT" == calibration ]]; then
    pcf1_require_safe_input "$shard_sandbox_probe"
    arguments+=(--shard-sandbox-probe "$shard_sandbox_probe")
  elif [[ -e "$shard_sandbox_probe" || -L "$shard_sandbox_probe" ]]; then
    pcf1_die "confirmation shard unexpectedly contains a sandbox probe"
  fi
done
python=$(pcf1_python)
pcf1_export_runtime
"$python" "$RUNTIME/pipeline/merge_pcf1_evaluation_shards.py" \
  --arm "$ARM" \
  --split "$SPLIT" \
  --data "$DATA" \
  --data-report "$DATA_REPORT" \
  --shard-root "$SHARD_ROOT" \
  "${arguments[@]}" \
  --candidates-output "$CANDIDATES_OUTPUT" \
  --report "$REPORT"
pcf1_validate_json "$REPORT" shohin-pcf1-merged-evaluation-v1
chmod a-w "$CANDIDATES_OUTPUT" "$REPORT"
