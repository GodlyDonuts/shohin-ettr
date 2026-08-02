#!/bin/bash
#SBATCH --job-name=shohin-terminal-measure
#SBATCH --partition=normal
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/terminal-measure-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/terminal-measure-%j.err

set -euo pipefail
umask 077

SELF_SHA256=${SELF_SHA256:?set submitted measurement script hash}
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
TERMINAL_RUN_DIR=${TERMINAL_RUN_DIR:?set completed terminal run}
OUTPUT=${OUTPUT:?set fresh evaluation output}
ROUTE_OUTPUT=${ROUTE_OUTPUT:?set fresh route receipt}
PLAN_OUTPUT=${PLAN_OUTPUT:?set fresh successor plan receipt}
CAMPAIGN_ROOT=${CAMPAIGN_ROOT:?set bounded campaign output root}
MEASUREMENT_SCRIPT=${MEASUREMENT_SCRIPT:?set exact measurement script}
MEASUREMENT_SCRIPT_SHA256=${MEASUREMENT_SCRIPT_SHA256:?set measurement script hash}
ROUTE_SCRIPT=${ROUTE_SCRIPT:?set exact route script}
ROUTE_SCRIPT_SHA256=${ROUTE_SCRIPT_SHA256:?set route script hash}
SUCCESSOR_SCRIPT=${SUCCESSOR_SCRIPT:?set exact successor script}
SUCCESSOR_SCRIPT_SHA256=${SUCCESSOR_SCRIPT_SHA256:?set successor script hash}
DATA_SEED=${DATA_SEED:-13}
MAX_BATCHES=${MAX_BATCHES:-32}
REQUIRED_DEVICE_CLASS=${REQUIRED_DEVICE_CLASS:-cuda}
PYTHON_ROOT=${PYTHON_ROOT:-/lustre/fs1/home/sa305415/shohin/miniforge3}
EVAL_TIME=${EVAL_TIME:-00:10:00}
EVAL_GRES=${EVAL_GRES:-gpu:1}
EVAL_EXCLUDE=${EVAL_EXCLUDE:-evc26,evc31,evc32,evc33,evc40,evc43,evc44}

for path in "$CODE_ROOT" "$RELEASE_ROOT" "$ETTR_DATA_ROOT" \
  "$TOKENIZER" "$PROTECTED_CHECKPOINT" "$JOINT_RUN_DIR" \
  "$COMPILER_RUN_DIR" "$TERMINAL_RUN_DIR" "$OUTPUT" \
  "$ROUTE_OUTPUT" "$PLAN_OUTPUT" "$CAMPAIGN_ROOT" \
  "$MEASUREMENT_SCRIPT" "$ROUTE_SCRIPT" "$SUCCESSOR_SCRIPT" \
  "$PYTHON_ROOT"; do
  [[ "$path" == /* ]] || {
    echo "terminal measurement paths must be absolute" >&2
    exit 2
  }
done
[[ -d "$CAMPAIGN_ROOT" && ! -L "$CAMPAIGN_ROOT" ]] || {
  echo "terminal measurement campaign root differs" >&2
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
    echo "terminal measurement script receipt differs" >&2
    exit 2
  }
done
[[ "$(sha256sum "$0" | cut -d ' ' -f 1)" == "$SELF_SHA256" \
  && "$SELF_SHA256" == "$MEASUREMENT_SCRIPT_SHA256" ]] || {
  echo "submitted terminal measurement script differs" >&2
  exit 2
}
cd "$CODE_ROOT"
[[ "$(tr -d '\r\n' < SOURCE_COMMIT)" == "$SOURCE_COMMIT" ]] \
  && [[ "$(sha256sum SHA256SUMS | cut -d ' ' -f 1)" \
    == "$RUNTIME_SHA256SUMS_SHA256" ]] \
  && [[ -z "$(sha256sum -c SHA256SUMS 2>&1 | grep -v ': OK$')" ]] \
  || {
    echo "terminal measurement runtime differs" >&2
    exit 2
  }
(
  cd "$TERMINAL_RUN_DIR"
  sha256sum -c SHA256SUMS
)
terminal_sha=$(sha256sum "$TERMINAL_RUN_DIR/SHA256SUMS" | cut -d ' ' -f 1)
for output in "$OUTPUT" "$ROUTE_OUTPUT" "$PLAN_OUTPUT"; do
  [[ ! -e "$output" && ! -L "$output" ]] || {
    echo "terminal measurement output exists" >&2
    exit 2
  }
done

eval_exports="ALL,CODE_ROOT=$CODE_ROOT,SOURCE_COMMIT=$SOURCE_COMMIT"
eval_exports+=",RUNTIME_SHA256SUMS_SHA256=$RUNTIME_SHA256SUMS_SHA256"
eval_exports+=",RELEASE_ROOT=$RELEASE_ROOT,RELEASE_SHA256=$RELEASE_SHA256"
eval_exports+=",ETTR_DATA_ROOT=$ETTR_DATA_ROOT,TOKENIZER=$TOKENIZER"
eval_exports+=",PROTECTED_CHECKPOINT=$PROTECTED_CHECKPOINT"
eval_exports+=",JOINT_RUN_DIR=$JOINT_RUN_DIR,COMPILER_RUN_DIR=$COMPILER_RUN_DIR"
eval_exports+=",TERMINAL_RUN_DIR=$TERMINAL_RUN_DIR"
eval_exports+=",TERMINAL_RUN_SHA256SUMS_SHA256=$terminal_sha"
eval_exports+=",OUTPUT=$OUTPUT,DATA_SEED=$DATA_SEED,MAX_BATCHES=$MAX_BATCHES"
eval_exports+=",REQUIRED_DEVICE_CLASS=$REQUIRED_DEVICE_CLASS"
eval_exports+=",PYTHON_ROOT=$PYTHON_ROOT"
eval_job=$(sbatch --parsable \
  --job-name=shohin-terminal-independent-eval \
  --partition=normal \
  --time="$EVAL_TIME" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=2 \
  --mem=48G \
  --gres="$EVAL_GRES" \
  --exclude="$EVAL_EXCLUDE" \
  --export="$eval_exports" \
  "$CODE_ROOT/train/jobs/eval_parallel_terminal_state.sbatch")
[[ "$eval_job" =~ ^[0-9]+$ ]] || {
  echo "terminal evaluation submission differs: $eval_job" >&2
  exit 2
}

route_exports="ALL,CODE_ROOT=$CODE_ROOT,SOURCE_COMMIT=$SOURCE_COMMIT"
route_exports+=",RUNTIME_SHA256SUMS_SHA256=$RUNTIME_SHA256SUMS_SHA256"
route_exports+=",TERMINAL_RUN_DIR=$TERMINAL_RUN_DIR"
route_exports+=",EVAL_OUTPUT=$OUTPUT,ROUTE_OUTPUT=$ROUTE_OUTPUT"
route_exports+=",PYTHON_ROOT=$PYTHON_ROOT"
route_job=$(sbatch --parsable \
  --dependency="afterok:$eval_job" \
  --kill-on-invalid-dep=yes \
  --export="$route_exports" \
  "$ROUTE_SCRIPT")
[[ "$route_job" =~ ^[0-9]+$ ]] || {
  scancel "$eval_job"
  echo "terminal route submission differs: $route_job" >&2
  exit 2
}

successor_exports="ALL,SELF_SHA256=$SUCCESSOR_SCRIPT_SHA256"
successor_exports+=",CODE_ROOT=$CODE_ROOT,SOURCE_COMMIT=$SOURCE_COMMIT"
successor_exports+=",RUNTIME_SHA256SUMS_SHA256=$RUNTIME_SHA256SUMS_SHA256"
successor_exports+=",RELEASE_ROOT=$RELEASE_ROOT,RELEASE_SHA256=$RELEASE_SHA256"
successor_exports+=",ETTR_DATA_ROOT=$ETTR_DATA_ROOT,TOKENIZER=$TOKENIZER"
successor_exports+=",PROTECTED_CHECKPOINT=$PROTECTED_CHECKPOINT"
successor_exports+=",JOINT_RUN_DIR=$JOINT_RUN_DIR,COMPILER_RUN_DIR=$COMPILER_RUN_DIR"
successor_exports+=",TERMINAL_RUN_DIR=$TERMINAL_RUN_DIR"
successor_exports+=",TERMINAL_RUN_SHA256SUMS_SHA256=$terminal_sha"
successor_exports+=",ROUTE_OUTPUT=$ROUTE_OUTPUT,PLAN_OUTPUT=$PLAN_OUTPUT"
successor_exports+=",CAMPAIGN_ROOT=$CAMPAIGN_ROOT"
successor_exports+=",MEASUREMENT_SCRIPT=$MEASUREMENT_SCRIPT"
successor_exports+=",MEASUREMENT_SCRIPT_SHA256=$MEASUREMENT_SCRIPT_SHA256"
successor_exports+=",ROUTE_SCRIPT=$ROUTE_SCRIPT"
successor_exports+=",ROUTE_SCRIPT_SHA256=$ROUTE_SCRIPT_SHA256"
successor_exports+=",SUCCESSOR_SCRIPT=$SUCCESSOR_SCRIPT"
successor_exports+=",SUCCESSOR_SCRIPT_SHA256=$SUCCESSOR_SCRIPT_SHA256"
successor_exports+=",PYTHON_ROOT=$PYTHON_ROOT"
successor_job=$(sbatch --parsable \
  --dependency="afterok:$route_job" \
  --kill-on-invalid-dep=yes \
  --export="$successor_exports" \
  "$SUCCESSOR_SCRIPT")
[[ "$successor_job" =~ ^[0-9]+$ ]] || {
  scancel "$eval_job" "$route_job"
  echo "terminal successor dispatch differs: $successor_job" >&2
  exit 2
}
printf 'evaluation_job=%s\nroute_job=%s\nsuccessor_dispatch_job=%s\n' \
  "$eval_job" "$route_job" "$successor_job"
