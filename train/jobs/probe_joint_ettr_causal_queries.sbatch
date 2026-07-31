#!/bin/bash
#SBATCH --job-name=shohin-joint-causal-probe
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc32,evc33,evc40,evc43,evc44
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/joint-causal-probe-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/joint-causal-probe-%j.err

set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set immutable source root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set exact source commit}
RUNTIME_SHA256SUMS_SHA256=${RUNTIME_SHA256SUMS_SHA256:?set runtime receipt hash}
RELEASE_ROOT=${RELEASE_ROOT:?set ETTR release root}
RELEASE_SHA256=${RELEASE_SHA256:?set ETTR release hash}
ETTR_DATA_ROOT=${ETTR_DATA_ROOT:?set ETTR source root}
TOKENIZER=${TOKENIZER:?set tokenizer path}
PROTECTED_CHECKPOINT=${PROTECTED_CHECKPOINT:?set protected checkpoint}
RUN_DIR=${RUN_DIR:?set completed joint run directory}
OUTPUT=${OUTPUT:?set fresh causal-query probe output}
PYTHON_ROOT=${PYTHON_ROOT:-/lustre/fs1/home/sa305415/shohin/miniforge3}
MAX_BATCHES=${MAX_BATCHES:-128}

cd "$CODE_ROOT"
[[ "$(tr -d '\r\n' < SOURCE_COMMIT)" == "$SOURCE_COMMIT" ]] \
  && [[ "$(sha256sum SHA256SUMS | awk '{print $1}')" \
    == "$RUNTIME_SHA256SUMS_SHA256" ]] \
  && [[ -z "$(sha256sum -c SHA256SUMS 2>&1 | grep -v ': OK$')" ]] \
  || {
    echo "joint causal-query probe runtime differs" >&2
    exit 2
  }
(
  cd "$RUN_DIR"
  sha256sum -c SHA256SUMS
)
run_contract_sha=$(sha256sum "$RUN_DIR/run-contract.json" | awk '{print $1}')
joint_model_sha=$(sha256sum "$RUN_DIR/joint-model-final.pt" | awk '{print $1}')
if [[ -e "$OUTPUT" || -L "$OUTPUT" ]]; then
  echo "joint causal-query probe output exists" >&2
  exit 2
fi
"$PYTHON_ROOT/bin/python" \
  "$CODE_ROOT/train/probe_joint_ettr_causal_queries.py" \
  --release-root "$RELEASE_ROOT" \
  --release-sha256 "$RELEASE_SHA256" \
  --data-root "$ETTR_DATA_ROOT" \
  --tokenizer "$TOKENIZER" \
  --protected-checkpoint "$PROTECTED_CHECKPOINT" \
  --run-contract "$RUN_DIR/run-contract.json" \
  --run-contract-sha256 "$run_contract_sha" \
  --joint-model "$RUN_DIR/joint-model-final.pt" \
  --joint-model-sha256 "$joint_model_sha" \
  --output "$OUTPUT" \
  --source-commit "$SOURCE_COMMIT" \
  --max-batches "$MAX_BATCHES"
