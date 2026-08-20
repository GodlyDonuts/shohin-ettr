#!/bin/bash
set -euo pipefail

required=(
  PYTHON RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE_COMMIT
  GPT_MODEL_ROOT OVERLAY_ROOT OVERLAY_REPORT MECHANICS_REPORT REVISION_CHECKPOINT
  PREPARED_ROOT ASSESSORS ASSESSORS_SHA256
  QWEN_MODEL_ROOT ALIGNED_CHECKPOINT COMMIT_CHECKPOINT
  ENVIRONMENT_RECEIPT ENVIRONMENT_RECEIPT_SHA256 ENVIRONMENT_TREE_SHA256
  CROSS_HOST_CONTRACT CROSS_HOST_CONTRACT_SHA256 RUN_ROOT
  REVISION_MARGIN_THRESHOLD REVISION_RELIABILITY_VETO
)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_verify_runtime
for path in "$GPT_MODEL_ROOT" "$OVERLAY_ROOT" "$QWEN_MODEL_ROOT" "$PREPARED_ROOT"; do
  [[ -d "$path" && ! -L "$path" ]]
done
for path in "$OVERLAY_REPORT" "$MECHANICS_REPORT" "$REVISION_CHECKPOINT" "$ASSESSORS" "$ALIGNED_CHECKPOINT" "$COMMIT_CHECKPOINT" "$ENVIRONMENT_RECEIPT" "$CROSS_HOST_CONTRACT" "$PREPARED_ROOT/receipt.json" "$PREPARED_ROOT/source.jsonl"; do
  [[ -f "$path" && ! -L "$path" ]]
done
[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]]
[[ "$RUN_ROOT" == /lustre/fs1/home/sa305415/shohin/artifacts/* ]]
readarray -t prepared < <("$PYTHON" -P -s -B - "$PREPARED_ROOT/receipt.json" <<'PY'
import json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    payload.get("schema") != "shohin-gpt-oss-120b-commit-confirmation-inputs-v1"
    or payload.get("status") != "complete"
    or payload.get("benchmark") != "mmlu_pro"
    or payload.get("rows") != 256
    or payload.get("excluded_identity_overlap") != 0
    or payload.get("prior_q36_external_identity_overlap") != 0
    or payload.get("prior_confirmation_identity_overlap") != 0
    or payload.get("assessor_access_count") != 0
):
    raise SystemExit("confirmation preparation receipt differs")
print(payload["source_output_sha256"])
PY
)
[[ ${#prepared[@]} -eq 1 ]]
source_sha256=${prepared[0]}
q36_verify_sha256 "$PREPARED_ROOT/source.jsonl" "$source_sha256"
q36_verify_sha256 "$CROSS_HOST_CONTRACT" "$CROSS_HOST_CONTRACT_SHA256"

mkdir -m 700 "$RUN_ROOT"
workdir=/lustre/fs1/home/sa305415/shohin
gpu_exclude=evc26,evc29,evc31,evc32,evc38,evc50
selector_exclude=evc26,evc29,evc31,evc32,evc33,evc34,evc37,evc38,evc43,evc46,evc50
candidate_root="$RUN_ROOT/candidates"
qwen_draft_root="$RUN_ROOT/qwen_drafts"
selection_root="$RUN_ROOT/selection"
score="$RUN_ROOT/score.json"
result="$RUN_ROOT/result.json"
gpt_common="PYTHON=$PYTHON,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,MODEL_ROOT=$GPT_MODEL_ROOT,OVERLAY_ROOT=$OVERLAY_ROOT,OVERLAY_REPORT=$OVERLAY_REPORT,MECHANICS_REPORT=$MECHANICS_REPORT,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

qwen_jobs=()
for shard in 0 1 2 3; do
  qwen_exports="RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,SOURCE_COMMIT=$SOURCE_COMMIT,PYTHON=$PYTHON,MODEL_ROOT=$QWEN_MODEL_ROOT,MODEL_MANIFEST=$QWEN_MODEL_ROOT/SHA256SUMS,MODEL_MANIFEST_SHA256=$Q36_MODEL_MANIFEST_SHA256,MODEL_REVISION=$Q36_MODEL_REVISION,MODEL_CONFIG_SHA256=$Q36_MODEL_CONFIG_SHA256,ENVIRONMENT_RECEIPT=$ENVIRONMENT_RECEIPT,ENVIRONMENT_RECEIPT_SHA256=$ENVIRONMENT_RECEIPT_SHA256,ENVIRONMENT_TREE_SHA256=$ENVIRONMENT_TREE_SHA256,EXTERNAL_SCRIPT=$RUNTIME/train/hf_q36_mtr_external_evaluate.py,EXTERNAL_SCRIPT_SHA256=$(q36_sha256 "$RUNTIME/train/hf_q36_mtr_external_evaluate.py"),ARM=unchanged,ADAPTER_CHECKPOINT=$ALIGNED_CHECKPOINT,SOURCE=$PREPARED_ROOT/source.jsonl,SOURCE_SHA256=$source_sha256,OUTPUT_ROOT=$qwen_draft_root,EXPECTED_ROWS=256,SHARD_COUNT=4,SHARD_INDEX=$shard,CONFIRMATION_MMLU_PRO=1"
  job=$(env -u SLURM_OVERLAP -u SLURM_WHOLE sbatch --parsable \
    --chdir="$workdir" --exclude="$selector_exclude" --export="$qwen_exports" \
    "$RUNTIME/train/jobs/q36_mtr_external_evaluate.sbatch")
  qwen_jobs+=("$job")
done
qwen_dependency=$(IFS=:; printf '%s' "${qwen_jobs[*]}")
draft_candidates=()
for shard in 00 01 02 03; do
  draft_candidates+=("$qwen_draft_root/unchanged/shard_$shard/candidates.jsonl")
done
draft_joined=$(IFS=:; printf '%s' "${draft_candidates[*]}")

eval_jobs=()
for arm in unchanged revision; do
  for shard in 0 1 2 3; do
    exports="$gpt_common,ARM=$arm,SHARD_INDEX=$shard,SOURCE=$PREPARED_ROOT/source.jsonl,EXPECTED_SOURCE_SHA256=$source_sha256,OUTPUT_ROOT=$candidate_root,CONFIRMATION_MMLU_PRO=1"
    dependency=()
    if [[ "$arm" == unchanged ]]; then
      exports="$exports,DRAFT_CANDIDATES=none"
    else
      exports="$exports,DRAFT_CANDIDATES=$draft_joined,REVISION_CHECKPOINT=$REVISION_CHECKPOINT"
      dependency=(--dependency="afterok:$qwen_dependency")
    fi
    job=$(env -u SLURM_OVERLAP -u SLURM_WHOLE sbatch --parsable \
      --chdir="$workdir" --exclude="$gpu_exclude" "${dependency[@]}" --export="$exports" \
      "$RUNTIME/train/jobs/gpt_oss_120b_evaluate.sbatch")
    eval_jobs+=("$job")
  done
done
eval_dependency=$(IFS=:; printf '%s' "${eval_jobs[*]}")
revision_paths=()
unchanged_paths=()
for shard in 00 01 02 03; do
  revision_paths+=("$candidate_root/revision/shard_$shard/candidates.jsonl")
  unchanged_paths+=("$candidate_root/unchanged/shard_$shard/candidates.jsonl")
done
revision_joined=$(IFS=:; printf '%s' "${revision_paths[*]}")
unchanged_joined=$(IFS=:; printf '%s' "${unchanged_paths[*]}")
selector_exports="RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,SOURCE_COMMIT=$SOURCE_COMMIT,PYTHON=$PYTHON,MODEL_ROOT=$QWEN_MODEL_ROOT,MODEL_MANIFEST=$QWEN_MODEL_ROOT/SHA256SUMS,MODEL_MANIFEST_SHA256=$Q36_MODEL_MANIFEST_SHA256,MODEL_REVISION=$Q36_MODEL_REVISION,MODEL_CONFIG_SHA256=$Q36_MODEL_CONFIG_SHA256,ALIGNED_CHECKPOINT=$ALIGNED_CHECKPOINT,COMMIT_CHECKPOINT=$COMMIT_CHECKPOINT,HOST=gpt_oss_120b_confirmation,SOURCE=$PREPARED_ROOT/source.jsonl,REVISION_CANDIDATES=$revision_joined,UNCHANGED_CANDIDATES=$unchanged_joined,OUTPUT=$selection_root/candidates.jsonl,SELECTIONS=$selection_root/selections.jsonl,REPORT=$selection_root/application.json,ENVIRONMENT_RECEIPT=$ENVIRONMENT_RECEIPT,ENVIRONMENT_RECEIPT_SHA256=$ENVIRONMENT_RECEIPT_SHA256,ENVIRONMENT_TREE_SHA256=$ENVIRONMENT_TREE_SHA256,CROSS_HOST_CONTRACT=$CROSS_HOST_CONTRACT,CROSS_HOST_CONTRACT_SHA256=$CROSS_HOST_CONTRACT_SHA256,RUN_ROOT=$RUN_ROOT,REVISION_MARGIN_THRESHOLD=$REVISION_MARGIN_THRESHOLD,REVISION_RELIABILITY_VETO=$REVISION_RELIABILITY_VETO"
selector_job=$(env -u SLURM_OVERLAP -u SLURM_WHOLE sbatch --parsable \
  --chdir="$workdir" --exclude="$selector_exclude" --dependency="afterok:$eval_dependency" \
  --export="$selector_exports" "$RUNTIME/train/jobs/q36_mtr_cross_host_commit.sbatch")
score_exports="PYTHON=$PYTHON,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,SOURCE=$PREPARED_ROOT/source.jsonl,EXPECTED_SOURCE_SHA256=$source_sha256,ASSESSORS=$ASSESSORS,EXPECTED_ASSESSORS_SHA256=$ASSESSORS_SHA256,CANDIDATE_ROOT=$candidate_root,OUTPUT=$score"
score_job=$(env -u SLURM_OVERLAP -u SLURM_WHOLE sbatch --parsable \
  --chdir="$workdir" --dependency="afterok:$eval_dependency" --export="$score_exports" \
  "$RUNTIME/pipeline/jobs/gpt_oss_120b_commit_confirmation_score.sbatch")
analysis_exports="RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,SOURCE_COMMIT=$SOURCE_COMMIT,PYTHON=$PYTHON,HOST=gpt_oss_120b_confirmation,SELECTIONS=$selection_root/selections.jsonl,APPLICATION_REPORT=$selection_root/application.json,SCORE=$score,OUTPUT=$result,CROSS_HOST_CONTRACT=$CROSS_HOST_CONTRACT,CROSS_HOST_CONTRACT_SHA256=$CROSS_HOST_CONTRACT_SHA256,RUN_ROOT=$RUN_ROOT,REVISION_MARGIN_THRESHOLD=$REVISION_MARGIN_THRESHOLD,REVISION_RELIABILITY_VETO=$REVISION_RELIABILITY_VETO"
analysis_job=$(env -u SLURM_OVERLAP -u SLURM_WHOLE sbatch --parsable \
  --chdir="$workdir" --dependency="afterok:$selector_job:$score_job" --export="$analysis_exports" \
  "$RUNTIME/train/jobs/q36_mtr_analyze_cross_host_commit.sbatch")

"$PYTHON" -P -s -B - "$RUN_ROOT/dispatch.json" "$selector_job" "$score_job" "$analysis_job" "${qwen_jobs[@]}" "${eval_jobs[@]}" <<'PY'
import json, os, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "shohin-gpt-oss-120b-commit-confirmation-dispatch-v1",
    "status": "submitted",
    "selector_job": int(sys.argv[2]),
    "score_job": int(sys.argv[3]),
    "analysis_job": int(sys.argv[4]),
    "qwen_fixed_draft_jobs": [int(value) for value in sys.argv[5:9]],
    "evaluation_jobs": [int(value) for value in sys.argv[9:]],
    "qwen_fixed_draft_job_count": 4,
    "evaluation_job_count": len(sys.argv[9:]),
    "independent_single_h100_jobs": len(sys.argv[5:]) + 1,
    "revision_margin_threshold": float(os.environ["REVISION_MARGIN_THRESHOLD"]),
    "revision_reliability_veto": os.environ["REVISION_RELIABILITY_VETO"],
    "requeue": False,
    "duplicate_scientific_jobs": 0,
    "assessor_access_phase": "dependent_cpu_score_after_generation",
}
temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
with temporary.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
PY
chmod a-w "$RUN_ROOT/dispatch.json"
printf 'gpt_oss_confirmation qwen=%s eval=%s selector=%s score=%s analysis=%s root=%s\n' "$qwen_dependency" "$eval_dependency" "$selector_job" "$score_job" "$analysis_job" "$RUN_ROOT"
