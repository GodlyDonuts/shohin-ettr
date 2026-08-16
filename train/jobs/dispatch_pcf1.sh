#!/bin/bash
# Submit exactly one held-root PCF1 graph, or print its immutable dry-run plan.

set -euo pipefail

readonly PARTITION=normal
readonly EXCLUDED_NODES=evc26,evc29,evc31,evc32,evc33,evc37,evc38,evc46
readonly SUBMISSION_PATH=/apps/slurm/current/bin:/usr/bin:/bin
readonly MODEL_REVISION_PIN=81eaece1948f3875421d9a45bc55487d10e2d894
readonly MODEL_ROOT_PIN=/lustre/fs1/home/sa305415/shohin/artifacts/external/ministral-3-8b-reasoning-2512-81eaece
# Sole historical-environment control-plane exception; never a data/output path.
readonly PYTHON_ENTRYPOINT_PIN=/lustre/fs1/home/sa305415/shohin/envs/product-reasoning-b3a3603-r2/bin/python
readonly PYTHON_RESOLVED_PIN=/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python3.13
readonly PYTHON_SHA256_PIN=051a031d827eab9778e982571db754662809164c8a3ec01e9beea1e1088123e0
readonly MIN_FREE_BYTES=$((128 * 1024 * 1024 * 1024))
readonly MIN_FREE_INODES=150000
readonly DRAFT_SHARDS=16
readonly EVAL_SHARDS=4
export PATH=$SUBMISSION_PATH

readonly -a STAGES=(
  prepare_inputs
  mechanics
  b1_train
  draft_generate
  draft_merge
  materialize
  revision_train
  calibration_revision_eval
  calibration_revision_merge
  calibration_unchanged_eval
  calibration_unchanged_merge
  calibration_pairs
  commit_train
  confirmation_revision_eval
  confirmation_revision_merge
  confirmation_unchanged_eval
  confirmation_unchanged_merge
  confirmation_self_refinement_eval
  confirmation_self_refinement_merge
  confirmation_pairs
  commit_apply
  precompute_custody
  prescore_accounting
  authorize_score
  commit_score
  normalize
  final_accounting
  compute_custody
  final_compare
)
readonly -a PRESCORE_PREDECESSORS=(
  prepare_inputs mechanics b1_train draft_generate draft_merge materialize
  revision_train calibration_revision_eval calibration_revision_merge
  calibration_unchanged_eval calibration_unchanged_merge calibration_pairs
  commit_train confirmation_revision_eval confirmation_revision_merge
  confirmation_unchanged_eval confirmation_unchanged_merge
  confirmation_self_refinement_eval confirmation_self_refinement_merge
  confirmation_pairs commit_apply precompute_custody
)
readonly -a FINAL_PREDECESSORS=(
  "${PRESCORE_PREDECESSORS[@]}"
  prescore_accounting authorize_score commit_score normalize
)

SUBMITTED_IDS=()

stage_deps() {
  case "$1" in
    prepare_inputs) ;;
    mechanics) printf '%s' prepare_inputs ;;
    b1_train) printf '%s' mechanics ;;
    draft_generate) printf '%s' b1_train ;;
    draft_merge) printf '%s' draft_generate ;;
    materialize) printf '%s' draft_merge ;;
    revision_train) printf '%s' materialize ;;
    calibration_revision_eval|calibration_unchanged_eval) printf '%s' revision_train ;;
    calibration_revision_merge) printf '%s' calibration_revision_eval ;;
    calibration_unchanged_merge) printf '%s' calibration_unchanged_eval ;;
    calibration_pairs) printf '%s' 'calibration_revision_merge calibration_unchanged_merge' ;;
    commit_train) printf '%s' calibration_pairs ;;
    confirmation_revision_eval|confirmation_unchanged_eval|confirmation_self_refinement_eval) printf '%s' commit_train ;;
    confirmation_revision_merge) printf '%s' confirmation_revision_eval ;;
    confirmation_unchanged_merge) printf '%s' confirmation_unchanged_eval ;;
    confirmation_self_refinement_merge) printf '%s' confirmation_self_refinement_eval ;;
    confirmation_pairs) printf '%s' 'confirmation_revision_merge confirmation_unchanged_merge' ;;
    commit_apply) printf '%s' 'confirmation_pairs commit_train' ;;
    precompute_custody) printf '%s' 'commit_apply confirmation_self_refinement_merge' ;;
    prescore_accounting) printf '%s' precompute_custody ;;
    authorize_score) printf '%s' prescore_accounting ;;
    commit_score) printf '%s' authorize_score ;;
    normalize) printf '%s' commit_score ;;
    final_accounting) printf '%s' normalize ;;
    compute_custody) printf '%s' final_accounting ;;
    final_compare) printf '%s' compute_custody ;;
    *) die "unknown stage: $1" ;;
  esac
}

stage_gpus() {
  case "$1" in
    mechanics|b1_train|draft_generate|revision_train|calibration_revision_eval|calibration_unchanged_eval|commit_train|confirmation_revision_eval|confirmation_unchanged_eval|confirmation_self_refinement_eval|commit_apply) printf '1' ;;
    *) printf '0' ;;
  esac
}

stage_tasks() {
  case "$1" in
    draft_generate) printf '%s' "$DRAFT_SHARDS" ;;
    calibration_revision_eval|calibration_unchanged_eval|confirmation_revision_eval|confirmation_unchanged_eval|confirmation_self_refinement_eval) printf '%s' "$EVAL_SHARDS" ;;
    *) printf '1' ;;
  esac
}

stage_script() {
  case "$1" in
    prepare_inputs) printf '%s' pipeline/jobs/pcf1_prepare_inputs.sbatch ;;
    mechanics) printf '%s' train/jobs/pcf1_mechanics.sbatch ;;
    b1_train) printf '%s' train/jobs/pcf1_train_b1.sbatch ;;
    draft_generate) printf '%s' train/jobs/pcf1_generate_drafts.sbatch ;;
    draft_merge) printf '%s' pipeline/jobs/pcf1_merge_drafts.sbatch ;;
    materialize) printf '%s' pipeline/jobs/pcf1_materialize_data.sbatch ;;
    revision_train) printf '%s' train/jobs/pcf1_train_revision.sbatch ;;
    calibration_revision_eval|calibration_unchanged_eval|confirmation_revision_eval|confirmation_unchanged_eval|confirmation_self_refinement_eval) printf '%s' train/jobs/pcf1_evaluate.sbatch ;;
    calibration_revision_merge|calibration_unchanged_merge|confirmation_revision_merge|confirmation_unchanged_merge|confirmation_self_refinement_merge) printf '%s' pipeline/jobs/pcf1_merge_evaluation.sbatch ;;
    calibration_pairs) printf '%s' pipeline/jobs/pcf1_build_commit_pairs.sbatch ;;
    commit_train) printf '%s' train/jobs/pcf1_train_commit.sbatch ;;
    confirmation_pairs) printf '%s' pipeline/jobs/pcf1_build_confirmation_pairs.sbatch ;;
    commit_apply) printf '%s' train/jobs/pcf1_apply_commit.sbatch ;;
    precompute_custody) printf '%s' pipeline/jobs/pcf1_build_precompute_custody.sbatch ;;
    prescore_accounting|final_accounting) printf '%s' pipeline/jobs/pcf1_capture_accounting.sbatch ;;
    authorize_score) printf '%s' pipeline/jobs/pcf1_authorize_score.sbatch ;;
    commit_score) printf '%s' pipeline/jobs/pcf1_score_commit.sbatch ;;
    normalize) printf '%s' pipeline/jobs/pcf1_normalize.sbatch ;;
    compute_custody) printf '%s' pipeline/jobs/pcf1_build_compute_custody.sbatch ;;
    final_compare) printf '%s' pipeline/jobs/pcf1_compare.sbatch ;;
    *) die "unknown stage script: $1" ;;
  esac
}

set_job_id() {
  printf -v "JOB_ID_$1" '%s' "$2"
  SUBMITTED_IDS+=("$2")
}

get_job_id() {
  local variable=JOB_ID_$1
  printf '%s' "${!variable:-}"
}

die() {
  printf 'pcf1-dispatch: %s\n' "$*" >&2
  exit 2
}

require() {
  local name=$1
  [[ -n "${!name:-}" ]] || die "$name is required"
}

reject_protected() {
  local folded
  folded=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
  local word
  for word in holdout product public; do
    [[ "$folded" != *"$word"* ]] || die "protected path is not admissible: $1"
  done
}

reject_ambient_scheduler_controls() {
  local name
  for name in $(env | sed -n 's/=.*//p'); do
    case "$name" in
      SBATCH_*|SLURM_*)
        die "ambient scheduler control is not admissible: $name"
        ;;
    esac
  done
}

dry_run() {
  printf '%s\n' 'PCF1_DRY_RUN_V1 submission=false mutation=false retry=false successor=false'
  printf 'partition=%s excluded=%s draft_shards=%s eval_shards=%s batch_size=2\n' \
    "$PARTITION" "$EXCLUDED_NODES" "$DRAFT_SHARDS" "$EVAL_SHARDS"
  printf '%s\n' 'gate=unchanged>=387+all_domains_nonzero;revision>=unchanged+65;revision>=self_refinement+39;revision_domain_deltas>=0;commit>=revision+13;commit_domain_deltas>=0;commit_retains_revision_and_unchanged_correct>=0.95;exact_1289_zero_truncation_zero_malformed;complete_custody'
  printf '%s\n' 'stage|gpus|array_tasks|dependencies'
  local stage
  for stage in "${STAGES[@]}"; do
    printf '%s|%s|%s|%s\n' "$stage" "$(stage_gpus "$stage")" \
      "$(stage_tasks "$stage")" "$(stage_deps "$stage")"
  done
  printf '%s\n' 'terminal=final_compare stop_after_gate=true automatic_retry=false automatic_successor=false'
}

validate_export_value() {
  local value=$1
  [[ "$value" != *$'\n'* && "$value" != *$'\r'* && "$value" != *$'\t'* ]] || \
    die "Slurm export value contains a control delimiter"
  local atom name atom_value
  local old_ifs=$IFS
  IFS=,
  for atom in $value; do
    name=${atom%%=*}
    atom_value=${atom#*=}
    [[ "$atom" == *=* && "$name" =~ ^[A-Z][A-Z0-9_]*$ && -n "$atom_value" ]] || \
      die "Slurm export atom differs"
  done
  IFS=$old_ifs
}

validate_export_atom_value() {
  local value=$1
  [[ "$value" != *','* && "$value" != *$'\n'* && "$value" != *$'\r'* && "$value" != *$'\t'* ]] || \
    die "Slurm export input contains a delimiter"
}

validate_gpu_exports() {
  local exports=$1 atom name value folded
  local old_ifs=$IFS
  IFS=,
  for atom in $exports; do
    name=${atom%%=*}
    value=${atom#*=}
    folded=$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')
    if [[ "$folded" =~ assessor|holdout|product|public ]]; then
      if [[ "$name" == PYTHON && "$value" == "$PYTHON_ENTRYPOINT_PIN" ]]; then
        continue
      fi
      IFS=$old_ifs
      die "GPU stage export crosses the sealed-data firewall: $name"
    fi
  done
  IFS=$old_ifs
}

live_preflight() {
  reject_ambient_scheduler_controls
  local command
  for command in git lfs squeue sinfo sbatch scontrol scancel sacct sha256sum realpath bwrap prlimit; do
    command -v "$command" >/dev/null || die "missing preflight command: $command"
  done
  for variable in REPOSITORY_ROOT RUNTIME RUNTIME_MANIFEST_SHA256 PYTHON MODEL_ROOT MODEL_REVISION PAIRS MATH_BANK SCIENCE_BANK CODE_BANK B1_DATA_LEGACY RUN_ROOT RUN_ID; do
    require "$variable"
    validate_export_atom_value "${!variable}"
  done
  [[ "$RUN_ID" =~ ^[a-z0-9][a-z0-9_-]{2,63}$ ]] || die "RUN_ID differs"
  [[ "$RUN_ROOT" == /* ]] || die "RUN_ROOT must be absolute"
  for variable in RUNTIME MODEL_ROOT RUN_ROOT; do
    reject_protected "${!variable}"
  done
  [[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || die "RUN_ROOT must be fresh"
  [[ "$MODEL_REVISION" == "$MODEL_REVISION_PIN" ]] || die "model revision differs"
  [[ "$(realpath "$MODEL_ROOT")" == "$MODEL_ROOT_PIN" ]] || die "model root differs"
  [[ "$PYTHON" == "$PYTHON_ENTRYPOINT_PIN" ]] || die "Python venv entrypoint differs"
  [[ "$(realpath "$PYTHON")" == "$PYTHON_RESOLVED_PIN" ]] || \
    die "Python resolved path differs"
  [[ "$(sha256sum "$PYTHON" | cut -d' ' -f1)" == "$PYTHON_SHA256_PIN" ]] || \
    die "Python hash differs"
  [[ "$(command -v bwrap)" == /usr/bin/bwrap ]] || die "bubblewrap path differs"
  [[ "$(sha256sum /usr/bin/bwrap | cut -d' ' -f1)" == eb767688b8224d8d3dbe1f8cb30ac3dff9ae8b02ff0452eaec9f94874d4e0011 ]] || \
    die "bubblewrap hash differs"
  [[ "$(command -v prlimit)" == /usr/bin/prlimit ]] || die "prlimit path differs"
  [[ "$(sha256sum /usr/bin/prlimit | cut -d' ' -f1)" == 2c1c7948498f2cb755d8c93ecf72c0651f5a5db23f79cc39cfa6727693d241d5 ]] || \
    die "prlimit hash differs"
  [[ -d "$REPOSITORY_ROOT/.git" || -f "$REPOSITORY_ROOT/.git" ]] || \
    die "repository root differs"
  [[ -z "$(git -C "$REPOSITORY_ROOT" status --porcelain=v1 --untracked-files=all)" ]] || \
    die "repository must be clean"
  local source_branch=codex/pcf1-ministral-publication-confirmation
  [[ "$(git -C "$REPOSITORY_ROOT" branch --show-current)" == "$source_branch" ]] || \
    die "publication branch differs"
  local source_commit runtime_commit remote_commit remote_ref
  source_commit=$(git -C "$REPOSITORY_ROOT" rev-parse HEAD)
  runtime_commit=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_commit"])' "$RUNTIME/runtime.json")
  [[ "$source_commit" == "$runtime_commit" ]] || die "runtime/source commit differs"
  [[ "$(git -C "$REPOSITORY_ROOT" remote get-url origin)" == https://github.com/GodlyDonuts/shohin-ettr.git ]] || \
    die "private origin differs"
  [[ "$(git -C "$REPOSITORY_ROOT" remote get-url public)" == https://github.com/GodlyDonuts/shohin.git ]] || \
    die "public fetch remote differs"
  [[ "$(git -C "$REPOSITORY_ROOT" remote get-url --push public)" == DISABLED_PUBLIC_REPO_DO_NOT_PUSH ]] || \
    die "public push remote is not disabled"
  read -r remote_commit remote_ref <<<"$(
    git -C "$REPOSITORY_ROOT" ls-remote --exit-code origin "refs/heads/$source_branch"
  )"
  [[ "$remote_ref" == "refs/heads/$source_branch" && "$remote_commit" == "$source_commit" ]] || \
    die "private publication branch is not pushed at the source commit"
  test -f "$RUNTIME/SHA256SUMS" || die "runtime manifest is missing"
  [[ "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" == "$RUNTIME_MANIFEST_SHA256" ]] || \
    die "runtime manifest hash differs"
  # shellcheck source=/dev/null
  source "$RUNTIME/train/jobs/pcf1_common.sh"
  pcf1_verify_runtime_membership
  local environment_probe_sha256
  environment_probe_sha256=$(
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
      PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
      PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline" \
      "$PYTHON" - "$RUNTIME" "$RUNTIME_MANIFEST_SHA256" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

from capture_pcf1_environment import environment_payload

payload = environment_payload(Path(sys.argv[1]), sys.argv[2])
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
print(hashlib.sha256(encoded).hexdigest())
PY
  )
  [[ "$environment_probe_sha256" =~ ^[0-9a-f]{64}$ ]] || \
    die "environment replay receipt differs"
  [[ "$(sha256sum "$MODEL_ROOT/SHA256SUMS" | cut -d' ' -f1)" == 46cc9203a18a414e08a53109662c3802b57c046896185ca9ab31875e8167cf1f ]] || \
    die "authoritative model manifest differs"
  [[ "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" == 5aae04beb9f2a9949eb1df870cf47ba292012a066bdcdcb115a9ac43425f8086 ]] || \
    die "authoritative model config differs"
  [[ "$(sha256sum "$MODEL_ROOT/SOURCE_REVISION" | cut -d' ' -f1)" == 3576c1bfaa0652940d12817ad3267ffe65645dc558ceb9a153ffb72f7211a982 ]] || \
    die "authoritative model revision receipt differs"
  pcf1_verify_model_tree "$MODEL_ROOT" "$MODEL_ROOT/SHA256SUMS" 58 35706515534
  PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline" "$PYTHON" - "$PAIRS" "$MATH_BANK" "$SCIENCE_BANK" "$CODE_BANK" "$B1_DATA_LEGACY" <<'PY'
import hashlib
from pathlib import Path
import sys
from build_pcf1_data import FROZEN_CUSTODY
from prepare_pcf1_inputs import B1_SHA256

def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()

pairs, *banks, b1 = sys.argv[1:]
if digest(pairs) != FROZEN_CUSTODY.pairs_sha256:
    raise SystemExit("PCF1 pair hash differs")
if frozenset(map(digest, banks)) != FROZEN_CUSTODY.bank_sha256s:
    raise SystemExit("PCF1 bank hashes differ")
if digest(b1) != B1_SHA256:
    raise SystemExit("PCF1 B1 hash differs")
PY
  local quota_target quota_raw quota_values
  quota_target=$(dirname "$RUN_ROOT")
  [[ -d "$quota_target" ]] || die "quota target parent is missing"
  quota_raw=$(lfs quota -u "${USER:?}" "$quota_target")
  export PCF1_QUOTA_RAW=$quota_raw
  quota_values=$("$PYTHON" - "$MIN_FREE_BYTES" "$MIN_FREE_INODES" <<'PY'
import os
import re
import sys

minimum_bytes, minimum_inodes = map(int, sys.argv[1:])
candidates = []
for line in os.environ["PCF1_QUOTA_RAW"].splitlines():
    values = [int(token.rstrip("*")) for token in line.split() if re.fullmatch(r"\d+\*?", token)]
    if len(values) >= 6:
        candidates.append(values[:6])
if len(candidates) != 1:
    raise SystemExit("PCF1 quota output geometry differs")
used_kib, _soft_kib, limit_kib, used_inodes, _soft_inodes, limit_inodes = candidates[0]
headroom_bytes = (limit_kib - used_kib) * 1024
headroom_inodes = limit_inodes - used_inodes
if headroom_bytes < minimum_bytes or headroom_inodes < minimum_inodes:
    raise SystemExit("PCF1 durable-storage headroom is unsafe")
print(used_kib, limit_kib, used_inodes, limit_inodes, headroom_bytes, headroom_inodes)
PY
)
  read -r PCF1_USED_KIB PCF1_LIMIT_KIB PCF1_USED_INODES PCF1_LIMIT_INODES PCF1_HEADROOM_BYTES PCF1_HEADROOM_INODES <<<"$quota_values"
  export PCF1_USED_KIB PCF1_LIMIT_KIB PCF1_USED_INODES PCF1_LIMIT_INODES
  export PCF1_HEADROOM_BYTES PCF1_HEADROOM_INODES
  local queue_raw scheduler_raw
  queue_raw=$(squeue -h -u "$USER" -o '%i|%T|%P|%N')
  [[ -z "$queue_raw" ]] || die "account already has queued/running jobs"
  scheduler_raw=$(sinfo -h -N -p "$PARTITION" -o '%N|%t|%G')
  export PCF1_SCHEDULER_RAW=$scheduler_raw
  "$PYTHON" - "$EXCLUDED_NODES" <<'PY'
import os
import sys

excluded = set(sys.argv[1].split(","))
capable = []
for line in os.environ["PCF1_SCHEDULER_RAW"].splitlines():
    fields = line.split("|", 2)
    if len(fields) != 3:
        raise SystemExit("PCF1 scheduler output geometry differs")
    node, state, gres = fields
    if node not in excluded and state.rstrip("*") in {"idle", "mix"} and "h100" in gres.casefold():
        capable.append(node)
if not capable:
    raise SystemExit("PCF1 has no admissible capable dense host")
PY
  export PCF1_QUEUE_RAW=$queue_raw PCF1_SOURCE_COMMIT=$source_commit
  export PCF1_SOURCE_BRANCH=$source_branch PCF1_REMOTE_BRANCH_COMMIT=$remote_commit
  export PCF1_ORIGIN_MAIN
  PCF1_ORIGIN_MAIN=$(git -C "$REPOSITORY_ROOT" rev-parse refs/remotes/origin/main)
  export PCF1_RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256
  export PCF1_ENVIRONMENT_PROBE_SHA256=$environment_probe_sha256
  PCF1_PREFLIGHT_JSON=$("$PYTHON" - "$MIN_FREE_BYTES" "$MIN_FREE_INODES" <<'PY'
import json
import os
import sys

minimum_bytes, minimum_inodes = map(int, sys.argv[1:])
print(json.dumps({
    "schema": "shohin-pcf1-live-preflight-v1",
    "status": "safe",
    "repository": {
        "source_commit": os.environ["PCF1_SOURCE_COMMIT"],
        "source_branch": os.environ["PCF1_SOURCE_BRANCH"],
        "remote_branch_commit": os.environ["PCF1_REMOTE_BRANCH_COMMIT"],
        "origin_main": os.environ["PCF1_ORIGIN_MAIN"],
        "clean": True,
        "public_push_disabled": True,
    },
    "runtime_manifest_sha256": os.environ["PCF1_RUNTIME_MANIFEST_SHA256"],
    "environment_receipt_sha256": os.environ["PCF1_ENVIRONMENT_PROBE_SHA256"],
    "quota": {
        "raw": os.environ["PCF1_QUOTA_RAW"],
        "used_kib": int(os.environ["PCF1_USED_KIB"]),
        "limit_kib": int(os.environ["PCF1_LIMIT_KIB"]),
        "used_inodes": int(os.environ["PCF1_USED_INODES"]),
        "limit_inodes": int(os.environ["PCF1_LIMIT_INODES"]),
        "headroom_bytes": int(os.environ["PCF1_HEADROOM_BYTES"]),
        "headroom_inodes": int(os.environ["PCF1_HEADROOM_INODES"]),
        "minimum_headroom_bytes": minimum_bytes,
        "minimum_headroom_inodes": minimum_inodes,
    },
    "scheduler": {
        "queue_raw": os.environ["PCF1_QUEUE_RAW"],
        "normal_nodes_raw": os.environ["PCF1_SCHEDULER_RAW"],
        "capable_host_present": True,
    },
}, sort_keys=True))
PY
)
  export PCF1_PREFLIGHT_JSON
}

dependency_option() {
  local stage=$1 dependency identifiers=() dependency_stage
  for dependency_stage in $(stage_deps "$stage"); do
    dependency=$(get_job_id "$dependency_stage")
    [[ -n "$dependency" ]] || die "missing predecessor job ID: $dependency_stage"
    identifiers+=("$dependency")
  done
  if ((${#identifiers[@]})); then
    local joined
    joined=$(IFS=:; printf '%s' "${identifiers[*]}")
    printf 'afterok:%s\n' "$joined"
  fi
}

submit_stage() {
  local stage=$1 exports=$2
  reject_ambient_scheduler_controls
  local script=$RUNTIME/$(stage_script "$stage")
  local -a command=(
    sbatch --parsable --no-requeue --nodes=1 --ntasks=1
    --partition="$PARTITION" --exclude="$EXCLUDED_NODES"
  )
  validate_export_value "$exports"
  [[ -f "$script" ]] || die "missing packaged stage script: $script"
  [[ "$(stage_gpus "$stage")" != 1 ]] || validate_gpu_exports "$exports"
  local dependency
  dependency=$(dependency_option "$stage")
  [[ -z "$dependency" ]] || command+=(--dependency="$dependency")
  local tasks
  tasks=$(stage_tasks "$stage")
  [[ "$(stage_gpus "$stage")" != 1 ]] || \
    command+=(--gres=gpu:nvidia_h100_pcie:1)
  if [[ "$tasks" != 1 ]]; then
    command+=(--array="0-$((tasks - 1))")
  fi
  [[ "$stage" != prepare_inputs ]] || command+=(--hold)
  command+=(--export="$exports" "$script")
  local result
  result=$("${command[@]}")
  result=${result%%;*}
  [[ "$result" =~ ^[0-9]+$ ]] || die "sbatch returned an invalid job ID for $stage"
  set_job_id "$stage" "$result"
}

write_dispatch_receipts() {
  local prescore=$RUN_ROOT/dispatch/prescore_dispatch.json
  local final=$RUN_ROOT/dispatch/dispatch.json
  local pairs=() stage
  for stage in "${STAGES[@]}"; do
    pairs+=("$stage=$(get_job_id "$stage")")
  done
  "$PYTHON" - "$RUN_ID" "$prescore" "$final" "${pairs[@]}" <<'PY'
import json
import os
from pathlib import Path
import shutil
import sys

run_id, prescore_path, final_path, *pairs = sys.argv[1:]
job_ids = dict(pair.split("=", 1) for pair in pairs)
stages = [
    "prepare_inputs", "mechanics", "b1_train", "draft_generate", "draft_merge",
    "materialize", "revision_train", "calibration_revision_eval",
    "calibration_revision_merge", "calibration_unchanged_eval",
    "calibration_unchanged_merge", "calibration_pairs", "commit_train",
    "confirmation_revision_eval", "confirmation_revision_merge",
    "confirmation_unchanged_eval", "confirmation_unchanged_merge",
    "confirmation_self_refinement_eval", "confirmation_self_refinement_merge",
    "confirmation_pairs", "commit_apply", "precompute_custody",
    "prescore_accounting", "authorize_score", "commit_score", "normalize",
    "final_accounting", "compute_custody", "final_compare",
]
prescore = stages[:22]
final = stages[:26]
arrays = {
    "draft_generate": 16,
    "calibration_revision_eval": 4,
    "calibration_unchanged_eval": 4,
    "confirmation_revision_eval": 4,
    "confirmation_unchanged_eval": 4,
    "confirmation_self_refinement_eval": 4,
}
gpu_stages = {
    "mechanics", "b1_train", "draft_generate", "revision_train",
    "calibration_revision_eval", "calibration_unchanged_eval", "commit_train",
    "confirmation_revision_eval", "confirmation_unchanged_eval",
    "confirmation_self_refinement_eval", "commit_apply",
}
dependencies = {
    "prepare_inputs": [], "mechanics": ["prepare_inputs"], "b1_train": ["mechanics"],
    "draft_generate": ["b1_train"], "draft_merge": ["draft_generate"],
    "materialize": ["draft_merge"], "revision_train": ["materialize"],
    "calibration_revision_eval": ["revision_train"],
    "calibration_revision_merge": ["calibration_revision_eval"],
    "calibration_unchanged_eval": ["revision_train"],
    "calibration_unchanged_merge": ["calibration_unchanged_eval"],
    "calibration_pairs": ["calibration_revision_merge", "calibration_unchanged_merge"],
    "commit_train": ["calibration_pairs"],
    "confirmation_revision_eval": ["commit_train"],
    "confirmation_revision_merge": ["confirmation_revision_eval"],
    "confirmation_unchanged_eval": ["commit_train"],
    "confirmation_unchanged_merge": ["confirmation_unchanged_eval"],
    "confirmation_self_refinement_eval": ["commit_train"],
    "confirmation_self_refinement_merge": ["confirmation_self_refinement_eval"],
    "confirmation_pairs": ["confirmation_revision_merge", "confirmation_unchanged_merge"],
    "commit_apply": ["confirmation_pairs", "commit_train"],
    "precompute_custody": ["commit_apply", "confirmation_self_refinement_merge"],
    "prescore_accounting": ["precompute_custody"],
    "authorize_score": ["prescore_accounting"], "commit_score": ["authorize_score"],
    "normalize": ["commit_score"], "final_accounting": ["normalize"],
    "compute_custody": ["final_accounting"], "final_compare": ["compute_custody"],
}
resources = {
    stage: {
        "gpus": int(stage in gpu_stages),
        "is_array": stage in arrays,
        "array_tasks": arrays.get(stage, 1),
    }
    for stage in stages
}
common = {
    "schema": "shohin-pcf1-dispatch-v1",
    "status": "submitted",
    "run_id": run_id,
    "partition": "normal",
    "excluded_nodes": ["evc26", "evc29", "evc31", "evc32", "evc33", "evc37", "evc38", "evc46"],
    "terminal_stage": "final_compare",
    "retry_authorized": False,
    "successor_authorized": False,
    "stop_after_gate": True,
    "frozen_geometry": {"draft_shards": 16, "evaluation_shards_per_arm": 4, "batch_size": 2},
    "live_preflight": json.loads(os.environ["PCF1_PREFLIGHT_JSON"]),
}
root = Path(prescore_path).parent
if root.exists() or root.is_symlink():
    raise SystemExit("refusing existing PCF1 dispatch root")
temporary = root.with_name(f".{root.name}.tmp.{os.getpid()}")
temporary.mkdir(parents=True)
try:
    for name, predecessors in (
        (Path(prescore_path).name, prescore), (Path(final_path).name, final)
    ):
        payload = {
            **common,
            "accounting_predecessors": predecessors,
            "job_ids": {stage: job_ids[stage] for stage in predecessors},
            "stage_resources": {stage: resources[stage] for stage in predecessors},
            "dependencies": {stage: dependencies[stage] for stage in predecessors},
        }
        path = temporary / name
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    directory = os.open(temporary, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    os.rename(temporary, root)
    parent = os.open(root.parent, os.O_RDONLY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
except BaseException:
    if temporary.exists():
        shutil.rmtree(temporary)
    raise
PY
  chmod -R a-w "$RUN_ROOT/dispatch"
}

submit_graph() {
  [[ "${PCF1_SUBMIT_ACK:-}" == ONE_PROSPECTIVE_GATE_ONLY ]] || \
    die "set PCF1_SUBMIT_ACK=ONE_PROSPECTIVE_GATE_ONLY to submit"
  live_preflight
  mkdir -p "$RUN_ROOT/logs"
  cd "$RUN_ROOT"
  local cancelled=false
  cleanup() {
    local status=$?
    if [[ "$cancelled" == false && "$status" != 0 && ${#SUBMITTED_IDS[@]} -gt 0 ]]; then
      scancel "${SUBMITTED_IDS[@]}" >/dev/null 2>&1 || true
      cancelled=true
    fi
    exit "$status"
  }
  trap cleanup EXIT INT TERM

  local base="PATH=$SUBMISSION_PATH,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,PYTHON=$PYTHON,MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION,PREPARE_ROOT=$RUN_ROOT/prepared"
  local prepared=$RUN_ROOT/prepared custodian=$RUN_ROOT/custodian
  local mechanics=$RUN_ROOT/mechanics b1=$RUN_ROOT/b1 drafts=$RUN_ROOT/drafts
  local merged=$RUN_ROOT/merged data=$RUN_ROOT/data revision=$RUN_ROOT/revision
  local evaluations=$RUN_ROOT/evaluations pairs=$RUN_ROOT/pairs commit=$RUN_ROOT/commit
  local application=$RUN_ROOT/application custody=$RUN_ROOT/custody
  local score=$RUN_ROOT/score normalized=$RUN_ROOT/normalized
  local sandbox_receipt=$mechanics/sandbox_receipt.json
  local score_consumption=$RUN_ROOT/score.score-authorization-consumed.json
  local prescore_dispatch=$RUN_ROOT/dispatch/prescore_dispatch.json
  local final_dispatch=$RUN_ROOT/dispatch/dispatch.json

  submit_stage prepare_inputs "PATH=$SUBMISSION_PATH,RUNTIME=$RUNTIME,RUNTIME_MANIFEST_SHA256=$RUNTIME_MANIFEST_SHA256,PYTHON=$PYTHON,MODEL_ROOT=$MODEL_ROOT,MODEL_REVISION=$MODEL_REVISION,PAIRS=$PAIRS,MATH_BANK=$MATH_BANK,SCIENCE_BANK=$SCIENCE_BANK,CODE_BANK=$CODE_BANK,B1_DATA_LEGACY=$B1_DATA_LEGACY,OUTPUT=$prepared,ASSESSOR_OUTPUT=$custodian/confirmation_assessors.jsonl,ASSESSOR_RECEIPT_OUTPUT=$custodian/confirmation_assessor_receipt.json,CPU_RECEIPT_OUTPUT=$custodian/prepare_receipt.json"
  submit_stage mechanics "$base,MECHANICS_ROOT=$mechanics"
  submit_stage b1_train "$base,MECHANICS_REPORT=$mechanics/report.json,SANDBOX_RECEIPT=$sandbox_receipt,COMPUTE_HOST_RECEIPT=$mechanics/compute_host_receipt.json,OUTPUT=$b1"
  submit_stage draft_generate "$base,B1_CHECKPOINT=$b1/checkpoint_0000256.pt,B1_REPORT=$b1/report.json,B1_RECEIPT=$b1/pcf1_receipt.json,DRAFT_ROOT=$drafts,DRAFT_SHARDS=$DRAFT_SHARDS"
  submit_stage draft_merge "$base,DRAFT_ROOT=$drafts,DRAFT_SHARDS=$DRAFT_SHARDS,DRAFTS_OUTPUT=$merged/drafts.jsonl,DRAFTS_REPORT=$merged/drafts_report.json"
  submit_stage materialize "$base,DRAFTS=$merged/drafts.jsonl,DRAFTS_REPORT=$merged/drafts_report.json,ASSESSOR_RECEIPT=$custodian/confirmation_assessor_receipt.json,OUTPUT=$data"
  submit_stage revision_train "$base,DATA=$data/revision_train.jsonl,DATA_REPORT=$data/materialization_report.json,B1_CHECKPOINT=$b1/checkpoint_0000256.pt,B1_RECEIPT=$b1/pcf1_receipt.json,OUTPUT=$revision"
  submit_stage calibration_revision_eval "$base,ARM=revision,SPLIT=calibration,DATA=$data/commit_train_eval.jsonl,DATA_REPORT=$data/materialization_report.json,ADAPTER_CHECKPOINT=$revision/checkpoint_0000256.pt,OUTPUT_ROOT=$evaluations/calibration_revision,EVAL_SHARDS=$EVAL_SHARDS,MECHANICS_REPORT=$mechanics/report.json,SANDBOX_RECEIPT=$sandbox_receipt"
  submit_stage calibration_revision_merge "$base,ARM=revision,SPLIT=calibration,DATA=$data/commit_train_eval.jsonl,DATA_REPORT=$data/materialization_report.json,SHARD_ROOT=$evaluations/calibration_revision,EVAL_SHARDS=$EVAL_SHARDS,CANDIDATES_OUTPUT=$merged/calibration_revision_candidates.jsonl,REPORT=$merged/calibration_revision_report.json"
  submit_stage calibration_unchanged_eval "$base,ARM=unchanged,SPLIT=calibration,DATA=$data/commit_train_eval.jsonl,DATA_REPORT=$data/materialization_report.json,ADAPTER_CHECKPOINT=$b1/checkpoint_0000256.pt,OUTPUT_ROOT=$evaluations/calibration_unchanged,EVAL_SHARDS=$EVAL_SHARDS,MECHANICS_REPORT=$mechanics/report.json,SANDBOX_RECEIPT=$sandbox_receipt"
  submit_stage calibration_unchanged_merge "$base,ARM=unchanged,SPLIT=calibration,DATA=$data/commit_train_eval.jsonl,DATA_REPORT=$data/materialization_report.json,SHARD_ROOT=$evaluations/calibration_unchanged,EVAL_SHARDS=$EVAL_SHARDS,CANDIDATES_OUTPUT=$merged/calibration_unchanged_candidates.jsonl,REPORT=$merged/calibration_unchanged_report.json"
  submit_stage calibration_pairs "$base,CALIBRATION_DATA=$data/commit_train_eval.jsonl,REVISION_REPORT=$merged/calibration_revision_report.json,REVISION_CANDIDATES=$merged/calibration_revision_candidates.jsonl,UNCHANGED_REPORT=$merged/calibration_unchanged_report.json,UNCHANGED_CANDIDATES=$merged/calibration_unchanged_candidates.jsonl,CANDIDATES_ROOT=$merged,OUTPUT=$pairs/calibration_pairs.jsonl,REPORT=$pairs/calibration_pairs_report.json"
  submit_stage commit_train "$base,B1_CHECKPOINT=$b1/checkpoint_0000256.pt,B1_RECEIPT=$b1/pcf1_receipt.json,PAIRS=$pairs/calibration_pairs.jsonl,PAIRS_REPORT=$pairs/calibration_pairs_report.json,OUTPUT=$commit"
  submit_stage confirmation_revision_eval "$base,ARM=revision,SPLIT=confirmation,DATA=$data/development_eval.jsonl,DATA_REPORT=$data/materialization_report.json,ADAPTER_CHECKPOINT=$revision/checkpoint_0000256.pt,OUTPUT_ROOT=$evaluations/confirmation_revision,EVAL_SHARDS=$EVAL_SHARDS"
  submit_stage confirmation_revision_merge "$base,ARM=revision,SPLIT=confirmation,DATA=$data/development_eval.jsonl,DATA_REPORT=$data/materialization_report.json,SHARD_ROOT=$evaluations/confirmation_revision,EVAL_SHARDS=$EVAL_SHARDS,CANDIDATES_OUTPUT=$merged/confirmation_revision_candidates.jsonl,REPORT=$merged/confirmation_revision_report.json"
  submit_stage confirmation_unchanged_eval "$base,ARM=unchanged,SPLIT=confirmation,DATA=$data/development_eval.jsonl,DATA_REPORT=$data/materialization_report.json,ADAPTER_CHECKPOINT=$b1/checkpoint_0000256.pt,OUTPUT_ROOT=$evaluations/confirmation_unchanged,EVAL_SHARDS=$EVAL_SHARDS"
  submit_stage confirmation_unchanged_merge "$base,ARM=unchanged,SPLIT=confirmation,DATA=$data/development_eval.jsonl,DATA_REPORT=$data/materialization_report.json,SHARD_ROOT=$evaluations/confirmation_unchanged,EVAL_SHARDS=$EVAL_SHARDS,CANDIDATES_OUTPUT=$merged/confirmation_unchanged_candidates.jsonl,REPORT=$merged/confirmation_unchanged_report.json"
  submit_stage confirmation_self_refinement_eval "$base,ARM=self_refinement,SPLIT=confirmation,DATA=$data/development_eval.jsonl,DATA_REPORT=$data/materialization_report.json,ADAPTER_CHECKPOINT=$b1/checkpoint_0000256.pt,OUTPUT_ROOT=$evaluations/confirmation_self_refinement,EVAL_SHARDS=$EVAL_SHARDS"
  submit_stage confirmation_self_refinement_merge "$base,ARM=self_refinement,SPLIT=confirmation,DATA=$data/development_eval.jsonl,DATA_REPORT=$data/materialization_report.json,SHARD_ROOT=$evaluations/confirmation_self_refinement,EVAL_SHARDS=$EVAL_SHARDS,CANDIDATES_OUTPUT=$merged/confirmation_self_refinement_candidates.jsonl,REPORT=$merged/confirmation_self_refinement_report.json"
  submit_stage confirmation_pairs "$base,CONFIRMATION_DATA=$data/development_eval.jsonl,REVISION_REPORT=$merged/confirmation_revision_report.json,REVISION_CANDIDATES=$merged/confirmation_revision_candidates.jsonl,UNCHANGED_REPORT=$merged/confirmation_unchanged_report.json,UNCHANGED_CANDIDATES=$merged/confirmation_unchanged_candidates.jsonl,CANDIDATES_ROOT=$merged,OUTPUT=$pairs/confirmation_pairs.jsonl,REPORT=$pairs/confirmation_pairs_report.json"
  submit_stage commit_apply "$base,B1_CHECKPOINT=$b1/checkpoint_0000256.pt,B1_RECEIPT=$b1/pcf1_receipt.json,COMMIT_CHECKPOINT=$commit/commit.pt,PAIRS=$pairs/confirmation_pairs.jsonl,PAIRS_REPORT=$pairs/confirmation_pairs_report.json,SELECTIONS=$application/selections.jsonl,REPORT=$application/report.json"
  submit_stage precompute_custody "$base,RUN_ID=$RUN_ID,SOURCE_FREEZE_REPORT=$prepared/sources/report.json,TRAIN_SOURCES=$prepared/sources/train_sources.jsonl,DEVELOPMENT_SOURCES=$prepared/sources/development_sources.jsonl,REFERENCE_PREFLIGHT_ROWS=$prepared/sources/mbpp_reference_preflight.jsonl,REFERENCE_SANDBOX_RECEIPT=$prepared/sources/reference_sandbox_receipt.json,MERGED_DRAFTS=$merged/drafts.jsonl,MERGED_DRAFTS_REPORT=$merged/drafts_report.json,REVISION_TRAINING_DATA=$data/revision_train.jsonl,CALIBRATION_DATA=$data/commit_train_eval.jsonl,CONFIRMATION_DATA=$data/development_eval.jsonl,CONFIRMATION_ASSESSOR_RECEIPT=$custodian/confirmation_assessor_receipt.json,DATA_REPORT=$data/materialization_report.json,CALIBRATION_PAIRS=$pairs/calibration_pairs.jsonl,CALIBRATION_PAIR_REPORT=$pairs/calibration_pairs_report.json,CONFIRMATION_PAIRS=$pairs/confirmation_pairs.jsonl,CONFIRMATION_PAIR_REPORT=$pairs/confirmation_pairs_report.json,CALIBRATION_REVISION_REPORT=$merged/calibration_revision_report.json,CALIBRATION_UNCHANGED_REPORT=$merged/calibration_unchanged_report.json,CALIBRATION_REVISION_SHARD_ROOT=$evaluations/calibration_revision,CALIBRATION_UNCHANGED_SHARD_ROOT=$evaluations/calibration_unchanged,REVISION_REPORT=$merged/confirmation_revision_report.json,UNCHANGED_REPORT=$merged/confirmation_unchanged_report.json,SELF_REFINEMENT_REPORT=$merged/confirmation_self_refinement_report.json,B1_CHECKPOINT=$b1/checkpoint_0000256.pt,B1_TRAINING_REPORT=$b1/report.json,REVISION_CHECKPOINT=$revision/checkpoint_0000256.pt,REVISION_TRAINING_REPORT=$revision/report.json,COMMIT_CHECKPOINT=$commit/commit.pt,COMMIT_TRAINING_REPORT=$commit/report.json,COMMIT_APPLICATION_REPORT=$application/report.json,CONFIRMATION_SELECTIONS=$application/selections.jsonl,MECHANICS_REPORT=$mechanics/report.json,COMPUTE_HOST_RECEIPT=$mechanics/compute_host_receipt.json,SANDBOX_RECEIPT=$sandbox_receipt,OUTPUT_ROOT=$custody"
  submit_stage prescore_accounting "$base,RUN_ID=$RUN_ID,DISPATCH_RECEIPT=$prescore_dispatch,OUTPUT=$RUN_ROOT/accounting/prescore.json"
  submit_stage authorize_score "$base,RUN_ID=$RUN_ID,CONFIRMATION_DATA=$data/development_eval.jsonl,CONFIRMATION_ASSESSOR_RECEIPT=$custodian/confirmation_assessor_receipt.json,REVISION_REPORT=$merged/confirmation_revision_report.json,REVISION_CANDIDATES=$merged/confirmation_revision_candidates.jsonl,UNCHANGED_REPORT=$merged/confirmation_unchanged_report.json,UNCHANGED_CANDIDATES=$merged/confirmation_unchanged_candidates.jsonl,SELF_REFINEMENT_REPORT=$merged/confirmation_self_refinement_report.json,SELF_REFINEMENT_CANDIDATES=$merged/confirmation_self_refinement_candidates.jsonl,CANDIDATES_ROOT=$merged,CONFIRMATION_PAIRS=$pairs/confirmation_pairs.jsonl,CONFIRMATION_PAIR_REPORT=$pairs/confirmation_pairs_report.json,CONFIRMATION_SELECTIONS=$application/selections.jsonl,COMMIT_APPLICATION_REPORT=$application/report.json,COMMIT_TRAINING_REPORT=$commit/report.json,MECHANICS_REPORT=$mechanics/report.json,SANDBOX_RECEIPT=$sandbox_receipt,DATA_CUSTODY=$custody/data_custody.json,MODEL_CUSTODY=$custody/model_custody.json,RUNTIME_CUSTODY=$custody/runtime_custody.json,PRESCORE_DISPATCH_RECEIPT=$prescore_dispatch,PRESCORE_ACCOUNTING_RECEIPT=$RUN_ROOT/accounting/prescore.json,SCORE_OUTPUT_ROOT=$score,OUTPUT=$RUN_ROOT/score_authorization.json"
  submit_stage commit_score "$base,CONFIRMATION_DATA=$data/development_eval.jsonl,CONFIRMATION_ASSESSORS=$custodian/confirmation_assessors.jsonl,CONFIRMATION_ASSESSOR_RECEIPT=$custodian/confirmation_assessor_receipt.json,DATA_REPORT=$data/materialization_report.json,REVISION_REPORT=$merged/confirmation_revision_report.json,REVISION_CANDIDATES=$merged/confirmation_revision_candidates.jsonl,UNCHANGED_REPORT=$merged/confirmation_unchanged_report.json,UNCHANGED_CANDIDATES=$merged/confirmation_unchanged_candidates.jsonl,SELF_REFINEMENT_REPORT=$merged/confirmation_self_refinement_report.json,SELF_REFINEMENT_CANDIDATES=$merged/confirmation_self_refinement_candidates.jsonl,CANDIDATES_ROOT=$merged,CONFIRMATION_PAIRS=$pairs/confirmation_pairs.jsonl,CONFIRMATION_PAIRS_REPORT=$pairs/confirmation_pairs_report.json,SELECTIONS=$application/selections.jsonl,APPLICATION_REPORT=$application/report.json,TRAINING_REPORT=$commit/report.json,MECHANICS_REPORT=$mechanics/report.json,SANDBOX_RECEIPT=$sandbox_receipt,DATA_CUSTODY=$custody/data_custody.json,MODEL_CUSTODY=$custody/model_custody.json,RUNTIME_CUSTODY=$custody/runtime_custody.json,PRESCORE_DISPATCH_RECEIPT=$prescore_dispatch,PRESCORE_ACCOUNTING_RECEIPT=$RUN_ROOT/accounting/prescore.json,PRESCORE_AUTHORIZATION=$RUN_ROOT/score_authorization.json,OUTPUT_ROOT=$score"
  submit_stage normalize "$base,LEARNED_COMMIT_REPORT=$score/report.json,REVISION_REPORT=$merged/confirmation_revision_report.json,UNCHANGED_REPORT=$merged/confirmation_unchanged_report.json,SELF_REFINEMENT_REPORT=$merged/confirmation_self_refinement_report.json,DATA_CUSTODY=$custody/data_custody.json,MODEL_CUSTODY=$custody/model_custody.json,RUNTIME_CUSTODY=$custody/runtime_custody.json,SCORE_CONSUMPTION=$score_consumption,OUTPUT_ROOT=$normalized"
  submit_stage final_accounting "$base,RUN_ID=$RUN_ID,DISPATCH_RECEIPT=$final_dispatch,OUTPUT=$RUN_ROOT/accounting/final.json"
  submit_stage compute_custody "$base,RUN_ID=$RUN_ID,DATA_CUSTODY=$custody/data_custody.json,MODEL_CUSTODY=$custody/model_custody.json,RUNTIME_CUSTODY=$custody/runtime_custody.json,SANDBOX_RECEIPT=$sandbox_receipt,SCORE_SANDBOX_PROBE=$score.sandbox-probe.json,NORMALIZED_ROOT=$normalized,SCORE_CONSUMPTION=$score_consumption,DISPATCH_RECEIPT=$final_dispatch,ACCOUNTING_RECEIPT=$RUN_ROOT/accounting/final.json,OUTPUT=$RUN_ROOT/compute_custody.json"
  submit_stage final_compare "$base,LEARNED_COMMIT_REPORT=$normalized/learned_commit.json,TRAINED_REVISION_REPORT=$normalized/trained_revision.json,UNCHANGED_REPORT=$normalized/unchanged.json,SELF_REFINEMENT_REPORT=$normalized/self_refinement.json,DATA_CUSTODY=$custody/data_custody.json,MODEL_CUSTODY=$custody/model_custody.json,RUNTIME_CUSTODY=$custody/runtime_custody.json,COMPUTE_CUSTODY=$RUN_ROOT/compute_custody.json,OUTPUT=$RUN_ROOT/final_comparison.json"

  write_dispatch_receipts
  scontrol release "$(get_job_id prepare_inputs)"
  trap - EXIT INT TERM
  printf 'PCF1 graph submitted once: run_id=%s root_job=%s terminal_job=%s\n' \
    "$RUN_ID" "$(get_job_id prepare_inputs)" "$(get_job_id final_compare)"
  printf '%s\n' 'No retry or automatic successor is authorized; stop after final_compare.'
}

mode=${1:---dry-run}
case "$mode" in
  --dry-run)
    (($# <= 1)) || die "--dry-run takes no arguments"
    dry_run
    ;;
  --preflight)
    (($# == 1)) || die "--preflight takes no arguments"
    live_preflight
    printf '%s\n' "$PCF1_PREFLIGHT_JSON"
    ;;
  --submit)
    (($# == 1)) || die "--submit takes no arguments"
    submit_graph
    ;;
  *) die "usage: dispatch_pcf1.sh [--dry-run|--preflight|--submit]" ;;
esac
