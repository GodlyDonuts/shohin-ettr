#!/bin/bash
set -euo pipefail

required=(
  PYTHON RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT OVERLAY_ROOT OVERLAY_REPORT
  MECHANICS_REPORT DATA SOURCE ASSESSORS DRAFT_CANDIDATES RUN_ROOT
)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
[[ -x "$PYTHON" ]]
for path in "$RUNTIME" "$MODEL_ROOT" "$OVERLAY_ROOT"; do
  [[ -d "$path" && ! -L "$path" ]]
done
for path in "$OVERLAY_REPORT" "$MECHANICS_REPORT" "$DATA" "$SOURCE" "$ASSESSORS"; do
  [[ -f "$path" && ! -L "$path" ]]
done
[[ "$RUN_ROOT" == /lustre/fs1/home/sa305415/shohin/artifacts/* ]]
[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]]
IFS=: read -r -a draft_paths <<< "$DRAFT_CANDIDATES"
[[ ${#draft_paths[@]} -eq 4 ]]
for path in "${draft_paths[@]}"; do
  [[ -f "$path" && ! -L "$path" ]]
done

source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_verify_runtime
"$PYTHON" -P -s -B - "$MECHANICS_REPORT" <<'PY'
import json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    payload.get("schema") != "shohin-gpt-oss-120b-one-h100-mechanics-v1"
    or payload.get("status") != "pass"
    or payload.get("scientific_result") is not False
    or payload.get("score_or_assessor_data_accessed") is not False
    or payload.get("checkpoint_restore_exact") is not True
):
    raise SystemExit("GPT-OSS mechanics authorization differs")
PY

mkdir -m 700 "$RUN_ROOT"
workdir=/lustre/fs1/home/sa305415/shohin
gpu_exclude=evc26,evc29,evc31,evc32,evc38,evc50
fit_output="$RUN_ROOT/training"
candidate_root="$RUN_ROOT/candidates"
score_output="$RUN_ROOT/score.json"
sandbox_receipt="$RUN_ROOT/score_sandbox.json"
common="PYTHON=$PYTHON,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,MODEL_ROOT=$MODEL_ROOT,OVERLAY_ROOT=$OVERLAY_ROOT,OVERLAY_REPORT=$OVERLAY_REPORT,MECHANICS_REPORT=$MECHANICS_REPORT"

fit_job=$(env -u SLURM_OVERLAP -u SLURM_WHOLE sbatch --parsable \
  --chdir="$workdir" \
  --exclude="$gpu_exclude" \
  --export="$common,DATA=$DATA,OUTPUT=$fit_output" \
  "$RUNTIME/train/jobs/gpt_oss_120b_train_revision.sbatch")

eval_jobs=()
for arm in unchanged self_refinement revision; do
  for shard in 0 1 2 3; do
    exports="$common,ARM=$arm,SHARD_INDEX=$shard,SOURCE=$SOURCE,OUTPUT_ROOT=$candidate_root"
    dependency=()
    if [[ "$arm" == unchanged ]]; then
      exports="$exports,DRAFT_CANDIDATES=none"
    elif [[ "$arm" == self_refinement ]]; then
      exports="$exports,DRAFT_CANDIDATES=$DRAFT_CANDIDATES"
    else
      exports="$exports,DRAFT_CANDIDATES=$DRAFT_CANDIDATES,REVISION_CHECKPOINT=$fit_output/checkpoint_0000256.pt"
      dependency=(--dependency="afterok:$fit_job")
    fi
    job=$(env -u SLURM_OVERLAP -u SLURM_WHOLE sbatch --parsable \
      --chdir="$workdir" \
      --exclude="$gpu_exclude" \
      "${dependency[@]}" \
      --export="$exports" \
      "$RUNTIME/train/jobs/gpt_oss_120b_evaluate.sbatch")
    eval_jobs+=("$job")
  done
done

dependency=$(IFS=:; printf '%s' "${eval_jobs[*]}")
score_job=$(env -u SLURM_OVERLAP -u SLURM_WHOLE sbatch --parsable \
  --chdir="$workdir" \
  --dependency="afterok:$dependency" \
  --export="PYTHON=$PYTHON,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,ASSESSORS=$ASSESSORS,CANDIDATE_ROOT=$candidate_root,OUTPUT=$score_output,SANDBOX_RECEIPT=$sandbox_receipt" \
  "$RUNTIME/pipeline/jobs/gpt_oss_120b_score.sbatch")

dispatch="$RUN_ROOT/dispatch.json"
"$PYTHON" -P -s -B - "$dispatch" "$fit_job" "$score_job" "${eval_jobs[@]}" <<'PY'
import json, os, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "shohin-gpt-oss-120b-fixed-draft-dispatch-v1",
    "status": "submitted",
    "fit_job": int(sys.argv[2]),
    "score_job": int(sys.argv[3]),
    "evaluation_jobs": [int(value) for value in sys.argv[4:]],
    "evaluation_job_count": len(sys.argv[4:]),
    "independent_single_h100_evaluations": True,
    "array_jobs": 0,
    "excluded_nodes": ["evc26", "evc29", "evc31", "evc32", "evc38", "evc50"],
    "requeue": False,
    "slurm_overlap_scrubbed": True,
    "slurm_whole_scrubbed": True,
}
temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
with temporary.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
PY
chmod a-w "$dispatch"
printf 'gpt_oss_120b fit=%s eval=%s score=%s run_root=%s\n' \
  "$fit_job" "$dependency" "$score_job" "$RUN_ROOT"
