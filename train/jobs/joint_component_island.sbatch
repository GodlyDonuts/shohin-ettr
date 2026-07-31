#!/bin/bash
#SBATCH --job-name=shohin-ettr-joint-component
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc32,evc33,evc40,evc43,evc44
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/joint-component-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/joint-component-%j.err

set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set immutable source root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set exact source commit}
RUNTIME_SHA256SUMS_SHA256=${RUNTIME_SHA256SUMS_SHA256:?set runtime receipt hash}
RELEASE_ROOT=${RELEASE_ROOT:?set ETTR release root}
RELEASE_SHA256=${RELEASE_SHA256:?set ETTR release hash}
ETTR_DATA_ROOT=${ETTR_DATA_ROOT:?set ETTR data root}
TOKENIZER=${TOKENIZER:?set tokenizer path}
PARENT_RUN_DIR=${PARENT_RUN_DIR:?set parent tri-stream run}
OUTPUT=${OUTPUT:?set fresh component output}
COMPONENT=${COMPONENT:?set compiler, reactor, or reader}
DATA_SEED=${DATA_SEED:?set data seed}
UPDATES=${UPDATES:-5000}
START_POSITION=${START_POSITION:-0}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
GRADIENT_CLIP=${GRADIENT_CLIP:-1.0}
EVAL_BATCHES=${EVAL_BATCHES:-32}
PYTHON_ROOT=${PYTHON_ROOT:-/lustre/fs1/home/sa305415/shohin/miniforge3}

cd "$CODE_ROOT"
[[ "$(tr -d '\r\n' < SOURCE_COMMIT)" == "$SOURCE_COMMIT" ]] \
  && [[ "$(sha256sum SHA256SUMS | awk '{print $1}')" \
    == "$RUNTIME_SHA256SUMS_SHA256" ]] \
  && [[ -z "$(sha256sum -c SHA256SUMS 2>&1 | grep -v ': OK$')" ]] \
  || {
    echo "joint component runtime differs" >&2
    exit 2
  }
(
  cd "$PARENT_RUN_DIR"
  sha256sum -c SHA256SUMS
)
if [[ -e "$OUTPUT" || -L "$OUTPUT" ]]; then
  echo "joint component output exists" >&2
  exit 2
fi

parent_contract_sha=$(sha256sum \
  "$PARENT_RUN_DIR/run-contract.json" | awk '{print $1}')
parent_model_sha=$(sha256sum \
  "$PARENT_RUN_DIR/joint-model-final.pt" | awk '{print $1}')

"$PYTHON_ROOT/bin/python" \
  "$CODE_ROOT/train/train_ettr_joint_component_island.py" \
  --component "$COMPONENT" \
  --release-root "$RELEASE_ROOT" \
  --release-sha256 "$RELEASE_SHA256" \
  --data-root "$ETTR_DATA_ROOT" \
  --tokenizer "$TOKENIZER" \
  --parent-joint-model "$PARENT_RUN_DIR/joint-model-final.pt" \
  --parent-joint-model-sha256 "$parent_model_sha" \
  --parent-run-contract "$PARENT_RUN_DIR/run-contract.json" \
  --parent-run-contract-sha256 "$parent_contract_sha" \
  --output "$OUTPUT" \
  --source-commit "$SOURCE_COMMIT" \
  --data-seed "$DATA_SEED" \
  --updates "$UPDATES" \
  --start-position "$START_POSITION" \
  --learning-rate "$LEARNING_RATE" \
  --gradient-clip "$GRADIENT_CLIP" \
  --eval-batches "$EVAL_BATCHES"
