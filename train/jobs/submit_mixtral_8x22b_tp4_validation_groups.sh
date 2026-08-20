#!/bin/bash
# Submit four independent TP4 validation groups as sixteen single-H100 requests.

set -euo pipefail

required=(
  PYTHON RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_MANIFEST
  MODEL_MANIFEST_SHA256 MECHANICS_REPORT REVISION_CHECKPOINT SOURCE
  DRAFT_CANDIDATES RUN_ROOT FIT_JOB_IDS
)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
for path in "$PYTHON" "$RUNTIME" "$MODEL_ROOT" "$MODEL_MANIFEST" \
  "$MECHANICS_REPORT" "$REVISION_CHECKPOINT" "$SOURCE"; do
  [[ "$path" == /* ]] || { printf 'path must be absolute: %s\n' "$path" >&2; exit 2; }
done
[[ -x "$PYTHON" ]]
[[ -d "$RUNTIME" && ! -L "$RUNTIME" ]]
[[ -d "$MODEL_ROOT" && ! -L "$MODEL_ROOT" ]]
[[ -f "$MODEL_MANIFEST" && ! -L "$MODEL_MANIFEST" ]]
[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]]
[[ "$RUNTIME_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$MODEL_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]

IFS=: read -r -a draft_paths <<< "$DRAFT_CANDIDATES"
[[ ${#draft_paths[@]} -eq 16 ]]
for path in "${draft_paths[@]}"; do
  [[ "$path" == /* && -f "$path" && ! -L "$path" ]]
done
dependency_args=()
if [[ "$FIT_JOB_IDS" != none ]]; then
  IFS=, read -r -a fit_jobs <<< "$FIT_JOB_IDS"
  [[ ${#fit_jobs[@]} -eq 4 ]]
  dependency=afterok
  for job in "${fit_jobs[@]}"; do
    [[ "$job" =~ ^[0-9]+$ ]]
    dependency+=":$job"
  done
  dependency_args=(--dependency="$dependency")
fi

mkdir -m 700 "$RUN_ROOT"
mkdir -m 700 "$RUN_ROOT/coordination"
for group in 0 1 2 3; do
  mkdir -m 700 "$RUN_ROOT/coordination/group_${group}"
done
worker="$RUNTIME/train/jobs/mixtral_8x22b_multinode_tp_evaluate_matched.sbatch"
[[ -f "$worker" && ! -L "$worker" ]]
base_exports="PYTHON=$PYTHON,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
base_exports+=",MODEL_ROOT=$MODEL_ROOT,MODEL_MANIFEST=$MODEL_MANIFEST"
base_exports+=",MODEL_MANIFEST_SHA256=$MODEL_MANIFEST_SHA256"
base_exports+=",MECHANICS_REPORT=$MECHANICS_REPORT,REVISION_CHECKPOINT=$REVISION_CHECKPOINT"
base_exports+=",SOURCE=$SOURCE,RUN_ROOT=$RUN_ROOT"
base_exports+=",EXPECTED_ROWS=1023,SHARD_COUNT=16,SHARD_GROUP_COUNT=4"

rank_node_pools=(
  "evc22,evc27,evc35,evc39,evc44"
  "evc23,evc28,evc36,evc40,evc45"
  "evc24,evc30,evc41,evc47,evc49"
  "evc25,evc31,evc42,evc43,evc46,evc48"
)
jobs=()
records=()
cleanup_partial_submission() {
  if (( ${#jobs[@]} > 0 && ${#jobs[@]} < 16 )); then
    scancel "${jobs[@]}" 2>/dev/null || true
  fi
}
trap cleanup_partial_submission EXIT
for group in 0 1 2 3; do
  first_draft=$((group * 4))
  group_drafts=$(IFS=:; echo "${draft_paths[*]:first_draft:4}")
  IFS=: read -r -a selected_drafts <<< "$group_drafts"
  [[ ${#selected_drafts[@]} -eq 4 ]]
  for offset in 0 1 2 3; do
    [[ "${selected_drafts[$offset]}" == "${draft_paths[$((first_draft + offset))]}" ]]
  done
  group_exports="$base_exports,DRAFT_CANDIDATES=$group_drafts"
  for rank in 0 1 2 3; do
    job=$(sbatch --parsable "${dependency_args[@]}" --time=06:00:00 \
      --nodelist="${rank_node_pools[$rank]}" \
      --export="$group_exports,SHARD_GROUP_INDEX=$group,WORLD_RANK=$rank" "$worker")
    [[ "$job" =~ ^[0-9]+$ ]]
    jobs+=("$job")
    records+=("group=$group rank=$rank job=$job")
  done
done
trap - EXIT
printf '%s\n' "${records[@]}" > "$RUN_ROOT/submitted_jobs.txt"
chmod 400 "$RUN_ROOT/submitted_jobs.txt"
printf 'mixtral_tp4_validation_groups jobs=%s run_root=%s\n' \
  "$(IFS=,; echo "${jobs[*]}")" "$RUN_ROOT"
