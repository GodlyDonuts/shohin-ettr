#!/bin/bash
#SBATCH -J diverge-vcr1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=256G
#SBATCH -t 06:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/diverge_vcr1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
SOURCE_COMMIT=${SOURCE_COMMIT:?set immutable SOURCE_COMMIT}
OUTPUT=${OUTPUT:?set isolated output directory}
ARM=${ARM:?set vcr1 or role_blind}
MODE=${MODE:-scientific}
DATA_SHA256=${DATA_SHA256:?set verified VCR1 train-board SHA256}
DEVELOPMENT_DATA_SHA256=${DEVELOPMENT_DATA_SHA256:?set verified VCR1 development-board SHA256}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL=$BASE/artifacts/external/smollm3-3b-a07cc9a
SOURCE_CHECKPOINT=$BASE/artifacts/product_reasoning/smollm3-3b-a07cc9a/baseline_late2_v12u1000_rollout_replay_u400_r1/checkpoint_0000400.pt
DATA=${DATA:-$BASE/artifacts/reasoning/diverge_vcr1/train_seed2026080603.jsonl}
DEVELOPMENT_DATA=${DEVELOPMENT_DATA:-$BASE/artifacts/reasoning/diverge_vcr1/development_seed2026080603.jsonl}

export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" logs
cd "$BASE"

test "$(sha256sum "$MODEL/config.json" | awk '{print $1}')" = \
  c72b1031274ff4626e434d0019e88e95a767460135db9ee492eb80652b786af1
test "$(sha256sum "$MODEL/model-00001-of-00002.safetensors" | awk '{print $1}')" = \
  e5eed4113a925264c33e3cdf76cd9f547c06a207ef2270914a7274a2bf685ffd
test "$(sha256sum "$MODEL/model-00002-of-00002.safetensors" | awk '{print $1}')" = \
  6c3ad90646457295723e4da5ee8afac47006e20407b122df5174693dd8d68a43
test "$(sha256sum "$MODEL/model.safetensors.index.json" | awk '{print $1}')" = \
  e3a7254c086c4a78f95ad5864f435d165ffed1ed8998326ce5107b4423c15b76
test "$(sha256sum "$MODEL/tokenizer.json" | awk '{print $1}')" = \
  7b6a500b662a34eb3f0374db856ba4ad7de4c81040571d78dc0d357238930005
test "$(sha256sum "$SOURCE_CHECKPOINT" | awk '{print $1}')" = \
  34c82454e0c53609bc1ac6a9f127437080e431f147e28ae63b4080c413d9a82e
test "$(sha256sum "$DATA" | awk '{print $1}')" = "$DATA_SHA256"
test "$(sha256sum "$DEVELOPMENT_DATA" | awk '{print $1}')" = "$DEVELOPMENT_DATA_SHA256"
test ! -e "$OUTPUT" || { echo "refusing existing output: $OUTPUT" >&2; exit 2; }
mkdir -p "$(dirname "$OUTPUT")"
trap 'test ! -e "$OUTPUT" || chmod -R a-w "$OUTPUT"' EXIT

echo "[diverge-vcr1] node=${SLURMD_NODENAME:-?} code=$CODE_ROOT commit=$SOURCE_COMMIT arm=$ARM mode=$MODE"
nvidia-smi -L
"$PYTHON" -m pytest -q \
  "$CODE_ROOT/train/test_diverge_vcr1_workspace.py" \
  "$CODE_ROOT/train/test_diverge_vcr1_data.py" \
  "$CODE_ROOT/train/test_diverge_vcr1_product.py"
"$PYTHON" -m py_compile \
  "$CODE_ROOT/train/diverge_vcr1_workspace.py" \
  "$CODE_ROOT/train/diverge_vcr1_data.py" \
  "$CODE_ROOT/train/diverge_vcr1_product.py" \
  "$CODE_ROOT/train/train_diverge_vcr1.py"

COMMON=(
  --model-root "$MODEL"
  --model-revision a07cc9a04f16550a088caea529712d1d335b0ac1
  --model-loader causal
  --source-checkpoint "$SOURCE_CHECKPOINT"
  --source-checkpoint-sha256 34c82454e0c53609bc1ac6a9f127437080e431f147e28ae63b4080c413d9a82e
  --data "$DATA"
  --data-sha256 "$DATA_SHA256"
  --development-data "$DEVELOPMENT_DATA"
  --development-data-sha256 "$DEVELOPMENT_DATA_SHA256"
  --output "$OUTPUT"
  --arm "$ARM"
  --max-sequence-length 4096
  --learning-rate 3e-4
  --workspace-width 384
  --workspace-slots 8
  --recurrent-steps 4
  --attention-heads 8
  --ff-multiplier 4
  --validity-weight 0.20
  --correction-margin-weight 0.10
  --correction-margin 0.25
  --seed 2026080603
  --data-seed 2026080603
)

if [[ "$MODE" == smoke ]]; then
  "$PYTHON" "$CODE_ROOT/train/train_diverge_vcr1.py" \
    "${COMMON[@]}" \
    --updates 2 --gradient-accumulation 1 --max-rows 4 \
    --development-probe-rows 2 --log-interval 1
elif [[ "$MODE" == scientific ]]; then
  "$PYTHON" "$CODE_ROOT/train/train_diverge_vcr1.py" \
    "${COMMON[@]}" \
    --updates 200 --gradient-accumulation 8 --max-rows 2800 \
    --development-probe-rows 32 --log-interval 10
else
  echo "unknown VCR1 mode: $MODE" >&2
  exit 2
fi

echo "[diverge-vcr1] completed arm=$ARM mode=$MODE"
