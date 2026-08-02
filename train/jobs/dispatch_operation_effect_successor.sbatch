#!/bin/bash
#SBATCH --job-name=shohin-effect-successor
#SBATCH --partition=normal
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/effect-successor-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/effect-successor-%j.err

set -euo pipefail
umask 077

SELF_SHA256=${SELF_SHA256:?set submitted successor script hash}
CODE_ROOT=${CODE_ROOT:?set immutable campaign runtime}
SOURCE_COMMIT=${SOURCE_COMMIT:?set exact campaign source}
RUNTIME_SHA256SUMS_SHA256=${RUNTIME_SHA256SUMS_SHA256:?set runtime receipt hash}
RELEASE_ROOT=${RELEASE_ROOT:?set ETTR release root}
RELEASE_SHA256=${RELEASE_SHA256:?set ETTR release hash}
ETTR_DATA_ROOT=${ETTR_DATA_ROOT:?set ETTR data root}
TOKENIZER=${TOKENIZER:?set tokenizer artifact}
PROTECTED_CHECKPOINT=${PROTECTED_CHECKPOINT:?set protected checkpoint}
JOINT_RUN_DIR=${JOINT_RUN_DIR:?set composed Shohin run}
COMPILER_RUN_DIR=${COMPILER_RUN_DIR:?set algebraic compiler run}
TERMINAL_RUN_DIR=${TERMINAL_RUN_DIR:?set measured terminal run}
TERMINAL_RUN_SHA256SUMS_SHA256=${TERMINAL_RUN_SHA256SUMS_SHA256:?set terminal receipt hash}
ROUTE_OUTPUT=${ROUTE_OUTPUT:?set measured route receipt}
PLAN_OUTPUT=${PLAN_OUTPUT:?set fresh successor plan receipt}
CAMPAIGN_ROOT=${CAMPAIGN_ROOT:?set bounded campaign output root}
MEASUREMENT_SCRIPT=${MEASUREMENT_SCRIPT:?set exact measurement script}
MEASUREMENT_SCRIPT_SHA256=${MEASUREMENT_SCRIPT_SHA256:?set measurement script hash}
ROUTE_SCRIPT=${ROUTE_SCRIPT:?set exact route script}
ROUTE_SCRIPT_SHA256=${ROUTE_SCRIPT_SHA256:?set route script hash}
SUCCESSOR_SCRIPT=${SUCCESSOR_SCRIPT:?set exact successor script}
SUCCESSOR_SCRIPT_SHA256=${SUCCESSOR_SCRIPT_SHA256:?set successor script hash}
PYTHON_ROOT=${PYTHON_ROOT:-/lustre/fs1/home/sa305415/shohin/miniforge3}
TRAIN_TIME=${TRAIN_TIME:-00:45:00}
TRAIN_GRES=${TRAIN_GRES:-gpu:1}
TRAIN_EXCLUDE=${TRAIN_EXCLUDE:-evc26,evc31,evc32,evc33,evc40,evc43,evc44}

for path in "$CODE_ROOT" "$RELEASE_ROOT" "$ETTR_DATA_ROOT" \
  "$TOKENIZER" "$PROTECTED_CHECKPOINT" "$JOINT_RUN_DIR" \
  "$COMPILER_RUN_DIR" "$TERMINAL_RUN_DIR" "$ROUTE_OUTPUT" \
  "$PLAN_OUTPUT" "$CAMPAIGN_ROOT" "$MEASUREMENT_SCRIPT" \
  "$ROUTE_SCRIPT" "$SUCCESSOR_SCRIPT" "$PYTHON_ROOT"; do
  [[ "$path" == /* ]] || {
    echo "operation successor paths must be absolute" >&2
    exit 2
  }
done
[[ -d "$CAMPAIGN_ROOT" && ! -L "$CAMPAIGN_ROOT" \
  && ! -e "$PLAN_OUTPUT" && ! -L "$PLAN_OUTPUT" \
  && -f "$ROUTE_OUTPUT" && ! -L "$ROUTE_OUTPUT" ]] || {
  echo "operation successor custody differs" >&2
  exit 2
}
for script_spec in \
  "$MEASUREMENT_SCRIPT:$MEASUREMENT_SCRIPT_SHA256" \
  "$ROUTE_SCRIPT:$ROUTE_SCRIPT_SHA256" \
  "$SUCCESSOR_SCRIPT:$SUCCESSOR_SCRIPT_SHA256"; do
  script=${script_spec%:*}
  expected=${script_spec##*:}
  [[ -f "$script" && ! -L "$script" \
    && "$(sha256sum "$script" | cut -d ' ' -f 1)" == "$expected" ]] || {
    echo "operation successor script receipt differs" >&2
    exit 2
  }
done
[[ "$(sha256sum "$0" | cut -d ' ' -f 1)" == "$SELF_SHA256" \
  && "$SELF_SHA256" == "$SUCCESSOR_SCRIPT_SHA256" ]] || {
  echo "submitted operation successor script differs" >&2
  exit 2
}
cd "$CODE_ROOT"
[[ "$(tr -d '\r\n' < SOURCE_COMMIT)" == "$SOURCE_COMMIT" ]] \
  && [[ "$(sha256sum SHA256SUMS | cut -d ' ' -f 1)" \
    == "$RUNTIME_SHA256SUMS_SHA256" ]] \
  && [[ -z "$(sha256sum -c SHA256SUMS 2>&1 | grep -v ': OK$')" ]] \
  || {
    echo "operation successor runtime differs" >&2
    exit 2
  }
(
  cd "$TERMINAL_RUN_DIR"
  [[ "$(sha256sum SHA256SUMS | cut -d ' ' -f 1)" \
    == "$TERMINAL_RUN_SHA256SUMS_SHA256" ]]
  sha256sum -c SHA256SUMS
)
"$PYTHON_ROOT/bin/python" \
  "$CODE_ROOT/train/plan_operation_effect_successor.py" \
  --route "$ROUTE_OUTPUT" \
  --terminal-contract "$TERMINAL_RUN_DIR/pilot-contract.json" \
  --output "$PLAN_OUTPUT"
chmod 400 "$PLAN_OUTPUT"

read -r action output_tag architecture_seed data_seed updates \
  family_gate family_island family_state_binding warm_start < <(
  "$PYTHON_ROOT/bin/python" - "$PLAN_OUTPUT" <<'PY'
import json
import sys

plan = json.load(open(sys.argv[1], encoding="utf-8"))
if plan["action"] == "stop":
    print("stop - 0 0 0 0 0 0 0")
else:
    print(
        plan["action"],
        plan["output_tag"],
        plan["architecture_seed"],
        plan["data_seed"],
        plan["updates"],
        int(plan["family_gate"]),
        int(plan["family_island"]),
        int(plan["family_state_binding"]),
        int(plan["warm_start"]),
    )
PY
)
if [[ "$action" == stop ]]; then
  printf 'campaign_stop_plan=%s\n' "$PLAN_OUTPUT"
  exit 0
fi
[[ "$action" == submit_scientific_successor \
  && "$output_tag" =~ ^v(18|19|20|21)-[a-z0-9-]+$ ]] || {
  echo "operation successor plan differs" >&2
  exit 2
}

successor_output="$CAMPAIGN_ROOT/$output_tag"
eval_output="$CAMPAIGN_ROOT/${output_tag}-eval-seed13"
next_route_output="$CAMPAIGN_ROOT/${output_tag}-route-seed13.json"
next_plan_output="$CAMPAIGN_ROOT/${output_tag}-successor-plan.json"
for output in "$successor_output" "$eval_output" \
  "$next_route_output" "$next_plan_output"; do
  [[ ! -e "$output" && ! -L "$output" ]] || {
    echo "operation successor output exists" >&2
    exit 2
  }
done

warm_dir=
warm_sha=
if [[ "$warm_start" == 1 ]]; then
  warm_dir=$TERMINAL_RUN_DIR
  warm_sha=$TERMINAL_RUN_SHA256SUMS_SHA256
elif [[ "$warm_start" != 0 ]]; then
  echo "operation successor warm-start setting differs" >&2
  exit 2
fi

fit_exports="ALL,CODE_ROOT=$CODE_ROOT,SOURCE_COMMIT=$SOURCE_COMMIT"
fit_exports+=",RUNTIME_SHA256SUMS_SHA256=$RUNTIME_SHA256SUMS_SHA256"
fit_exports+=",RELEASE_ROOT=$RELEASE_ROOT,RELEASE_SHA256=$RELEASE_SHA256"
fit_exports+=",ETTR_DATA_ROOT=$ETTR_DATA_ROOT,TOKENIZER=$TOKENIZER"
fit_exports+=",PROTECTED_CHECKPOINT=$PROTECTED_CHECKPOINT"
fit_exports+=",JOINT_RUN_DIR=$JOINT_RUN_DIR,COMPILER_RUN_DIR=$COMPILER_RUN_DIR"
fit_exports+=",TERMINAL_WARM_START_DIR=$warm_dir"
fit_exports+=",TERMINAL_WARM_START_SHA256SUMS_SHA256=$warm_sha"
fit_exports+=",OUTPUT=$successor_output,CAUSAL_DELTA_WEIGHT=1"
fit_exports+=",ARCHITECTURE_SEED=$architecture_seed,DATA_SEED=$data_seed"
fit_exports+=",UPDATES=$updates,START_POSITION=0,LEARNING_RATE=3e-4"
fit_exports+=",GRADIENT_CLIP=1,TRAINING_INITIAL_STATE=oracle,EVAL_BATCHES=32"
fit_exports+=",WIDTH=512,LAYERS=4,NUM_HEADS=8,RELATION_WIDTH=64"
fit_exports+=",REQUIRED_DEVICE_CLASS=cuda,ATOMIC_EDITS=1"
fit_exports+=",ATOMIC_ACTION_WEIGHT=1,LEXICAL_COMMAND=1"
fit_exports+=",TOKEN_NATIVE_COMMAND_MASK=1"
fit_exports+=",TOKEN_NATIVE_COVER_VERIFIED_COMMAND_MASK=1"
fit_exports+=",TOKEN_NATIVE_SYNTAX_GRAPH_COMMAND=1"
fit_exports+=",TOKEN_NATIVE_DECLARATION_BINDING_COMMAND=1"
fit_exports+=",TOKEN_NATIVE_OPERATION_RECURRENCE_COMMAND=1"
fit_exports+=",TOKEN_NATIVE_OPERATION_STATE_COMMAND=1"
fit_exports+=",OPERATION_EFFECT_ROLE_ANCHORS=1"
fit_exports+=",OPERATION_EFFECT_WRITE_LINK_RAILS=1"
fit_exports+=",OPERATION_EFFECT_FAMILY_GATE=$family_gate"
fit_exports+=",OPERATION_EFFECT_FAMILY_ISLAND=$family_island"
fit_exports+=",OPERATION_EFFECT_FAMILY_STATE_BINDING=$family_state_binding"
fit_exports+=",PYTHON_ROOT=$PYTHON_ROOT"
fit_job=$(sbatch --parsable \
  --job-name="shohin-${output_tag%%-u1000*}" \
  --partition=normal \
  --time="$TRAIN_TIME" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem=64G \
  --gres="$TRAIN_GRES" \
  --exclude="$TRAIN_EXCLUDE" \
  --export="$fit_exports" \
  "$CODE_ROOT/train/jobs/parallel_terminal_state_pilot.sbatch")
[[ "$fit_job" =~ ^[0-9]+$ ]] || {
  echo "operation successor fit submission differs: $fit_job" >&2
  exit 2
}

measure_exports="ALL,SELF_SHA256=$MEASUREMENT_SCRIPT_SHA256"
measure_exports+=",CODE_ROOT=$CODE_ROOT,SOURCE_COMMIT=$SOURCE_COMMIT"
measure_exports+=",RUNTIME_SHA256SUMS_SHA256=$RUNTIME_SHA256SUMS_SHA256"
measure_exports+=",RELEASE_ROOT=$RELEASE_ROOT,RELEASE_SHA256=$RELEASE_SHA256"
measure_exports+=",ETTR_DATA_ROOT=$ETTR_DATA_ROOT,TOKENIZER=$TOKENIZER"
measure_exports+=",PROTECTED_CHECKPOINT=$PROTECTED_CHECKPOINT"
measure_exports+=",JOINT_RUN_DIR=$JOINT_RUN_DIR,COMPILER_RUN_DIR=$COMPILER_RUN_DIR"
measure_exports+=",TERMINAL_RUN_DIR=$successor_output"
measure_exports+=",OUTPUT=$eval_output,ROUTE_OUTPUT=$next_route_output"
measure_exports+=",PLAN_OUTPUT=$next_plan_output,CAMPAIGN_ROOT=$CAMPAIGN_ROOT"
measure_exports+=",MEASUREMENT_SCRIPT=$MEASUREMENT_SCRIPT"
measure_exports+=",MEASUREMENT_SCRIPT_SHA256=$MEASUREMENT_SCRIPT_SHA256"
measure_exports+=",ROUTE_SCRIPT=$ROUTE_SCRIPT"
measure_exports+=",ROUTE_SCRIPT_SHA256=$ROUTE_SCRIPT_SHA256"
measure_exports+=",SUCCESSOR_SCRIPT=$SUCCESSOR_SCRIPT"
measure_exports+=",SUCCESSOR_SCRIPT_SHA256=$SUCCESSOR_SCRIPT_SHA256"
measure_exports+=",PYTHON_ROOT=$PYTHON_ROOT"
measure_job=$(sbatch --parsable \
  --dependency="afterok:$fit_job" \
  --kill-on-invalid-dep=yes \
  --export="$measure_exports" \
  "$MEASUREMENT_SCRIPT")
[[ "$measure_job" =~ ^[0-9]+$ ]] || {
  scancel "$fit_job"
  echo "operation successor measurement submission differs: $measure_job" >&2
  exit 2
}
printf 'successor_fit_job=%s\nmeasurement_dispatch_job=%s\nplan=%s\n' \
  "$fit_job" "$measure_job" "$PLAN_OUTPUT"
