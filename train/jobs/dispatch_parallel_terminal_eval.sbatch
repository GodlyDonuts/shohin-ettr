#!/bin/bash
#SBATCH --job-name=shohin-terminal-eval-dispatch
#SBATCH --partition=normal
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/terminal-eval-dispatch-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/terminal-eval-dispatch-%j.err

set -euo pipefail
umask 077

SELF_SHA256=${SELF_SHA256:?set submitted dispatch script hash}
CODE_ROOT=${CODE_ROOT:?set immutable evaluation runtime}
SOURCE_COMMIT=${SOURCE_COMMIT:?set exact evaluation source}
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
ROUTE_SCRIPT=${ROUTE_SCRIPT:?set exact route batch script}
ROUTE_SCRIPT_SHA256=${ROUTE_SCRIPT_SHA256:?set route batch script hash}
DATA_SEED=${DATA_SEED:?set independent evaluation seed}
MAX_BATCHES=${MAX_BATCHES:-32}
REQUIRED_DEVICE_CLASS=${REQUIRED_DEVICE_CLASS:-cuda}
PYTHON_ROOT=${PYTHON_ROOT:-/lustre/fs1/home/sa305415/shohin/miniforge3}
EVAL_TIME=${EVAL_TIME:-00:10:00}
EVAL_GRES=${EVAL_GRES:-gpu:1}
EVAL_EXCLUDE=${EVAL_EXCLUDE:-evc26,evc31,evc32,evc33,evc40,evc43,evc44}

[[ "$(sha256sum "$0" | cut -d ' ' -f 1)" == "$SELF_SHA256" ]] || {
  echo "terminal evaluation dispatch script differs" >&2
  exit 2
}
for path in "$CODE_ROOT" "$RELEASE_ROOT" "$ETTR_DATA_ROOT" \
  "$TOKENIZER" "$PROTECTED_CHECKPOINT" "$JOINT_RUN_DIR" \
  "$COMPILER_RUN_DIR" "$TERMINAL_RUN_DIR" "$OUTPUT" \
  "$ROUTE_OUTPUT" "$ROUTE_SCRIPT" "$PYTHON_ROOT"; do
  [[ "$path" == /* ]] || {
    echo "terminal evaluation dispatch paths must be absolute" >&2
    exit 2
  }
done
[[ -f "$ROUTE_SCRIPT" && ! -L "$ROUTE_SCRIPT" \
  && "$(sha256sum "$ROUTE_SCRIPT" | cut -d ' ' -f 1)" \
    == "$ROUTE_SCRIPT_SHA256" ]] || {
  echo "terminal evaluation route script differs" >&2
  exit 2
}
cd "$CODE_ROOT"
[[ "$(tr -d '\r\n' < SOURCE_COMMIT)" == "$SOURCE_COMMIT" ]] \
  && [[ "$(sha256sum SHA256SUMS | cut -d ' ' -f 1)" \
    == "$RUNTIME_SHA256SUMS_SHA256" ]] \
  && [[ -z "$(sha256sum -c SHA256SUMS 2>&1 | grep -v ': OK$')" ]] \
  || {
    echo "terminal evaluation dispatch runtime differs" >&2
    exit 2
  }
(
  cd "$TERMINAL_RUN_DIR"
  sha256sum -c SHA256SUMS
)
terminal_sha=$(sha256sum "$TERMINAL_RUN_DIR/SHA256SUMS" | cut -d ' ' -f 1)
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" \
  && ! -e "$ROUTE_OUTPUT" && ! -L "$ROUTE_OUTPUT" ]] || {
  echo "terminal evaluation dispatch output exists" >&2
  exit 2
}

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
  --job-name=shohin-v15-independent-eval \
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
printf 'evaluation_job=%s\nroute_job=%s\nterminal_sha256s_sha256=%s\n' \
  "$eval_job" "$route_job" "$terminal_sha"
