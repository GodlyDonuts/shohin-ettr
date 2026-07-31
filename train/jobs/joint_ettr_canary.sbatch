#!/bin/bash
#SBATCH --job-name=shohin-ettr-joint
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc32,evc33,evc40,evc43,evc44
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/joint-ettr-canary-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/joint-ettr-canary-%j.err

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
GENERAL_SHARD_ROOT=${GENERAL_SHARD_ROOT:?set historical shard root}
OUTDIR=${OUTDIR:?set fresh output directory}
INITIALIZATION=${INITIALIZATION:?set warm or random}
UPDATES=${UPDATES:?set bounded update count}
GENERAL_POSITION_WEIGHT=${GENERAL_POSITION_WEIGHT:?set position weight}
ETTR_POSITION_WEIGHT=${ETTR_POSITION_WEIGHT:?set position weight}
ARCHITECTURE_SEED=${ARCHITECTURE_SEED:?set architecture seed}
BASE_SEED=${BASE_SEED:?set base seed}
DATA_SEED=${DATA_SEED:?set data seed}
PYTHON_ROOT=${PYTHON_ROOT:-/lustre/fs1/home/sa305415/shohin/miniforge3}
WARMUP_UPDATES=${WARMUP_UPDATES:-333}
TOTAL_UPDATES=${TOTAL_UPDATES:-300000}
QUERY_BINDING_WEIGHT=${QUERY_BINDING_WEIGHT:-1}
NLL_GRADIENT_CAP=${NLL_GRADIENT_CAP:-4}
GRADIENT_CLIP_MODE=${GRADIENT_CLIP_MODE:-global}

for path in \
  "$CODE_ROOT" \
  "$RELEASE_ROOT" \
  "$ETTR_DATA_ROOT" \
  "$TOKENIZER" \
  "$PROTECTED_CHECKPOINT" \
  "$GENERAL_SHARD_ROOT" \
  "$OUTDIR" \
  "$PYTHON_ROOT"; do
  [[ "$path" == /* ]] || {
    echo "joint canary paths must be absolute" >&2
    exit 2
  }
done
if [[ ! "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ \
  || ! "$RUNTIME_SHA256SUMS_SHA256" =~ ^[0-9a-f]{64}$ \
  || ! "$RELEASE_SHA256" =~ ^[0-9a-f]{64}$ \
  || "$INITIALIZATION" != warm && "$INITIALIZATION" != random \
  || ! "$UPDATES" =~ ^[1-9][0-9]*$ \
  || ! "$GENERAL_POSITION_WEIGHT" =~ ^[1-9][0-9]*$ \
  || ! "$ETTR_POSITION_WEIGHT" =~ ^[1-9][0-9]*$ \
  || ! "$ARCHITECTURE_SEED" =~ ^[0-9]+$ \
  || ! "$BASE_SEED" =~ ^[0-9]+$ \
  || ! "$DATA_SEED" =~ ^[0-9]+$ ]]; then
  echo "joint canary launch identity differs" >&2
  exit 2
fi
if [[ -e "$OUTDIR" || -L "$OUTDIR" ]]; then
  echo "joint canary output already exists" >&2
  exit 2
fi

cd "$CODE_ROOT"
[[ -r SOURCE_COMMIT ]] \
  && [[ "$(tr -d '\r\n' < SOURCE_COMMIT)" == "$SOURCE_COMMIT" ]] \
  || {
    echo "joint canary source commit differs" >&2
    exit 2
  }
[[ "$(sha256sum SHA256SUMS | awk '{print $1}')" \
  == "$RUNTIME_SHA256SUMS_SHA256" ]] \
  && [[ -z "$(sha256sum -c SHA256SUMS 2>&1 | grep -v ': OK$')" ]] \
  || {
    echo "joint canary runtime differs" >&2
    exit 2
  }
[[ "$(sha256sum "$RELEASE_ROOT/release.json" | awk '{print $1}')" \
  == "$RELEASE_SHA256" ]] || {
  echo "joint canary release differs" >&2
  exit 2
}
for name in finemath4 openwebmath code_python finemath3; do
  [[ -d "$GENERAL_SHARD_ROOT/$name" ]] || {
    echo "legacy general shard directory is missing" >&2
    exit 2
  }
done

"$PYTHON_ROOT/bin/python" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit(75)
if "H100" not in torch.cuda.get_device_name(0).upper():
    sys.exit(76)
torch.ones(1, device="cuda").add_(1)
torch.cuda.synchronize()
PY

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2
args=(
  --release-root "$RELEASE_ROOT"
  --release-sha256 "$RELEASE_SHA256"
  --ettr-data-root "$ETTR_DATA_ROOT"
  --tokenizer "$TOKENIZER"
  --protected-checkpoint "$PROTECTED_CHECKPOINT"
  --output "$OUTDIR"
  --source-commit "$SOURCE_COMMIT"
  --initialization "$INITIALIZATION"
  --updates "$UPDATES"
  --general-batch-size 16
  --general-position-weight "$GENERAL_POSITION_WEIGHT"
  --ettr-position-weight "$ETTR_POSITION_WEIGHT"
  --architecture-seed "$ARCHITECTURE_SEED"
  --base-seed "$BASE_SEED"
  --data-seed "$DATA_SEED"
  --total-updates "$TOTAL_UPDATES"
  --warmup-updates "$WARMUP_UPDATES"
  --nll-gradient-cap "$NLL_GRADIENT_CAP"
  --query-binding-weight "$QUERY_BINDING_WEIGHT"
  --gradient-clip-mode "$GRADIENT_CLIP_MODE"
)
for name in finemath4 openwebmath code_python finemath3; do
  args+=(--legacy-general-shard-dir "$GENERAL_SHARD_ROOT/$name")
  args+=(--legacy-general-weight 1)
done
"$PYTHON_ROOT/bin/python" \
  "$CODE_ROOT/train/train_ettr_joint_stream_canary.py" \
  "${args[@]}"

(
  cd "$OUTDIR"
  sha256sum \
    base-eval-final.pt \
    final-report.json \
    joint-model-final.pt \
    run-contract.json \
    train.jsonl \
    > SHA256SUMS
  chmod 400 SHA256SUMS
  sha256sum -c SHA256SUMS
)
