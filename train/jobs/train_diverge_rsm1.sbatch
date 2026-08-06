#!/bin/bash
#SBATCH -J diverge-rsm1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=256G
#SBATCH -t 06:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/diverge_rsm1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
SOURCE_COMMIT=${SOURCE_COMMIT:?set immutable SOURCE_COMMIT}
OUTPUT=${OUTPUT:?set isolated output directory}
MODE=${MODE:-component}
DATA_SHA256=${DATA_SHA256:?set verified CRP1 train-board SHA256}
DEVELOPMENT_DATA_SHA256=${DEVELOPMENT_DATA_SHA256:?set verified CRP1 development-board SHA256}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL=$BASE/artifacts/external/smollm3-3b-a07cc9a
SOURCE_CHECKPOINT=$BASE/artifacts/product_reasoning/smollm3-3b-a07cc9a/baseline_late2_v12u1000_rollout_replay_u400_r1/checkpoint_0000400.pt
CRP_CHECKPOINT=$BASE/artifacts/reasoning/diverge_crp1/train_guarded_scientific_43d81d0/checkpoint_0000200.pt
DATA=${DATA:-$BASE/artifacts/reasoning/diverge_crp1/train_seed2026080604.jsonl}
DEVELOPMENT_DATA=${DEVELOPMENT_DATA:-$BASE/artifacts/reasoning/diverge_crp1/development_seed2026080604.jsonl}

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
test "$(sha256sum "$SOURCE_CHECKPOINT" | awk '{print $1}')" = \
  34c82454e0c53609bc1ac6a9f127437080e431f147e28ae63b4080c413d9a82e
test "$(sha256sum "$CRP_CHECKPOINT" | awk '{print $1}')" = \
  588dce4f608fde47516a8b29feedc40bf7ee58d2ff2aa8b344848915dcacb5ce
test "$(sha256sum "$DATA" | awk '{print $1}')" = "$DATA_SHA256"
test "$(sha256sum "$DEVELOPMENT_DATA" | awk '{print $1}')" = "$DEVELOPMENT_DATA_SHA256"
test ! -e "$OUTPUT" || { echo "refusing existing output: $OUTPUT" >&2; exit 2; }
mkdir -p "$(dirname "$OUTPUT")"
trap 'test ! -e "$OUTPUT" || chmod -R a-w "$OUTPUT"' EXIT

echo "[diverge-rsm1] node=${SLURMD_NODENAME:-?} code=$CODE_ROOT commit=$SOURCE_COMMIT mode=$MODE"
nvidia-smi -L
"$PYTHON" -m py_compile \
  "$CODE_ROOT/train/diverge_rsm1_data.py" \
  "$CODE_ROOT/train/diverge_rsm1_workspace.py" \
  "$CODE_ROOT/train/diverge_rsm1_product.py" \
  "$CODE_ROOT/train/train_diverge_rsm1.py"
"$PYTHON" "$CODE_ROOT/train/test_diverge_rsm1_data.py"
"$PYTHON" "$CODE_ROOT/train/test_diverge_rsm1_workspace.py"
"$PYTHON" "$CODE_ROOT/train/test_diverge_rsm1_product.py"

COMMON=(
  --model-root "$MODEL"
  --model-revision a07cc9a04f16550a088caea529712d1d335b0ac1
  --model-loader causal
  --source-checkpoint "$SOURCE_CHECKPOINT"
  --source-checkpoint-sha256 34c82454e0c53609bc1ac6a9f127437080e431f147e28ae63b4080c413d9a82e
  --crp-checkpoint "$CRP_CHECKPOINT"
  --crp-checkpoint-sha256 588dce4f608fde47516a8b29feedc40bf7ee58d2ff2aa8b344848915dcacb5ce
  --packet-arm guarded
  --data "$DATA" --data-sha256 "$DATA_SHA256"
  --development-data "$DEVELOPMENT_DATA"
  --development-data-sha256 "$DEVELOPMENT_DATA_SHA256"
  --output "$OUTPUT"
  --state-width 256 --state-slots 24 --packet-slots 6
  --max-trace-steps 12 --attention-heads 8 --ff-multiplier 4
  --max-sequence-length 4096 --learning-rate 3e-4
  --seed 2026080605 --data-seed 2026080605
)

if [[ "$MODE" == smoke ]]; then
  "$PYTHON" "$CODE_ROOT/train/train_diverge_rsm1.py" \
    "${COMMON[@]}" --updates 2 --pairs-per-microbatch 4 \
    --gradient-accumulation 1 --max-rows 4 \
    --development-probe-rows 4 --log-interval 1
elif [[ "$MODE" == component ]]; then
  "$PYTHON" "$CODE_ROOT/train/train_diverge_rsm1.py" \
    "${COMMON[@]}" --updates 1600 --pairs-per-microbatch 4 \
    --gradient-accumulation 2 --max-rows 4800 \
    --development-probe-rows 96 --log-interval 20
else
  echo "unknown RSM1 mode: $MODE" >&2
  exit 2
fi

echo "[diverge-rsm1] completed mode=$MODE"
