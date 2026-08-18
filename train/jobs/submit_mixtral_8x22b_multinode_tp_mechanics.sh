#!/bin/bash
# Submit four independent one-H100 requests for one tensor-parallel mechanics run.

set -euo pipefail

required=(
  PYTHON RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_MANIFEST
  MODEL_MANIFEST_SHA256 RUN_ROOT
)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
for path in "$PYTHON" "$RUNTIME" "$MODEL_ROOT" "$MODEL_MANIFEST" "$RUN_ROOT"; do
  [[ "$path" == /* ]] || { printf 'path must be absolute: %s\n' "$path" >&2; exit 2; }
done
[[ -x "$PYTHON" ]]
[[ -d "$RUNTIME" && ! -L "$RUNTIME" ]]
[[ -d "$MODEL_ROOT" && ! -L "$MODEL_ROOT" ]]
[[ -f "$MODEL_MANIFEST" && ! -L "$MODEL_MANIFEST" ]]
[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]]
[[ "$RUNTIME_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$MODEL_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]

mkdir -m 700 "$RUN_ROOT"
mkdir -m 700 "$RUN_ROOT/coordination"
worker="$RUNTIME/train/jobs/mixtral_8x22b_multinode_tp_mechanics.sbatch"
[[ -f "$worker" && ! -L "$worker" ]]
exports="PYTHON=$PYTHON,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256"
exports+=",MODEL_ROOT=$MODEL_ROOT,MODEL_MANIFEST=$MODEL_MANIFEST"
exports+=",MODEL_MANIFEST_SHA256=$MODEL_MANIFEST_SHA256,RUN_ROOT=$RUN_ROOT"

jobs=()
# Disjoint pools keep the four independent one-H100 allocations on distinct
# hosts while letting Slurm admit each rank opportunistically.
rank_node_pools=(
  "evc22,evc27,evc35,evc39,evc44"
  "evc23,evc28,evc36,evc40,evc45"
  "evc24,evc30,evc41,evc47,evc49"
  "evc25,evc31,evc42,evc43,evc46,evc48"
)
cleanup_partial_submission() {
  if (( ${#jobs[@]} > 0 && ${#jobs[@]} < 4 )); then
    scancel "${jobs[@]}" 2>/dev/null || true
  fi
}
trap cleanup_partial_submission EXIT
for rank in 0 1 2 3; do
  job=$(sbatch --parsable --nodelist="${rank_node_pools[$rank]}" \
    --export="$exports,WORLD_RANK=$rank" "$worker")
  [[ "$job" =~ ^[0-9]+$ ]]
  jobs+=("$job")
done
trap - EXIT
printf '%s\n' "${jobs[@]}" > "$RUN_ROOT/submitted_jobs.txt"
chmod 400 "$RUN_ROOT/submitted_jobs.txt"
printf 'mixtral_multinode_tp_mechanics jobs=%s,%s,%s,%s run_root=%s\n' \
  "${jobs[0]}" "${jobs[1]}" "${jobs[2]}" "${jobs[3]}" "$RUN_ROOT"
