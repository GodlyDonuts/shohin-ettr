#!/bin/bash
#SBATCH --job-name=pcf1-draft-merge
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc33,evc37,evc38,evc46
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/pcf1_draft_merge_%j.out
#SBATCH --error=logs/pcf1_draft_merge_%j.err

set -euo pipefail

: "${RUNTIME:?}" "${RUNTIME_MANIFEST_SHA256:?}"
test -f "$RUNTIME/SHA256SUMS"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
source "$RUNTIME/train/jobs/pcf1_common.sh"
pcf1_verify_runtime
pcf1_bind_prepared_inputs
for variable in DRAFT_ROOT DRAFT_SHARDS DRAFTS_OUTPUT DRAFTS_REPORT; do
  pcf1_require "$variable"
done
[[ "$DRAFT_SHARDS" == 16 ]] || pcf1_die "DRAFT_SHARDS must equal frozen value 16"
pcf1_require_fresh "$DRAFTS_OUTPUT"
pcf1_require_fresh "$DRAFTS_REPORT"
pcf1_reject_protected_path "$SOURCE_ROOT"
pcf1_require_file "$SOURCE_ROOT/report.json"
arguments=()
for ((index = 0; index < DRAFT_SHARDS; index++)); do
  shard_report=$DRAFT_ROOT/shards/$index/report.json
  shard_candidates=$DRAFT_ROOT/shards/$index/candidates.jsonl
  pcf1_require_safe_input "$shard_report"
  pcf1_require_safe_input "$shard_candidates"
  arguments+=(--shard-report "$shard_report" --shard-candidates "$shard_candidates")
done
python=$(pcf1_python)
pcf1_export_runtime
"$python" "$RUNTIME/pipeline/merge_pcf1_draft_shards.py" \
  --source-root "$SOURCE_ROOT" \
  "${arguments[@]}" \
  --output "$DRAFTS_OUTPUT" \
  --report "$DRAFTS_REPORT"
pcf1_validate_json "$DRAFTS_REPORT" shohin-pcf1-merged-drafts-v1
chmod a-w "$DRAFTS_OUTPUT" "$DRAFTS_REPORT"
