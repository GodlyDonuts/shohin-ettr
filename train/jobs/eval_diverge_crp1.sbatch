#!/bin/bash
#SBATCH -J crp1-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=192G
#SBATCH -t 06:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/diverge_crp1_eval_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
SOURCE_COMMIT=${SOURCE_COMMIT:?set immutable SOURCE_COMMIT}
OUTPUT=${OUTPUT:?set isolated report output}
ARM=${ARM:?set plain, guarded, or unguarded}
VARIANT=${VARIANT:?set wrong or correct}
DATA_SHA256=${DATA_SHA256:?set verified CRP1 evaluation-board SHA256}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL=$BASE/artifacts/external/smollm3-3b-a07cc9a
SOURCE_CHECKPOINT=$BASE/artifacts/product_reasoning/smollm3-3b-a07cc9a/baseline_late2_v12u1000_rollout_replay_u400_r1/checkpoint_0000400.pt
DATA=${DATA:-$BASE/artifacts/reasoning/diverge_crp1/evaluation_seed2026080604.jsonl}
ABLATION=${ABLATION:-normal}
COUNT=${COUNT:-480}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-384}
STAGE_MODEL=${STAGE_MODEL:-true}

export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" logs "$(dirname "$OUTPUT")"
cd "$BASE"

test "$(sha256sum "$MODEL/config.json" | awk '{print $1}')" = \
  c72b1031274ff4626e434d0019e88e95a767460135db9ee492eb80652b786af1
test "$(sha256sum "$MODEL/model-00001-of-00002.safetensors" | awk '{print $1}')" = \
  e5eed4113a925264c33e3cdf76cd9f547c06a207ef2270914a7274a2bf685ffd
test "$(sha256sum "$MODEL/model-00002-of-00002.safetensors" | awk '{print $1}')" = \
  6c3ad90646457295723e4da5ee8afac47006e20407b122df5174693dd8d68a43
test "$(sha256sum "$SOURCE_CHECKPOINT" | awk '{print $1}')" = \
  34c82454e0c53609bc1ac6a9f127437080e431f147e28ae63b4080c413d9a82e
test "$(sha256sum "$DATA" | awk '{print $1}')" = "$DATA_SHA256"
test ! -e "$OUTPUT" || { echo "refusing existing output: $OUTPUT" >&2; exit 2; }
trap 'test ! -e "$OUTPUT" || chmod a-w "$OUTPUT"' EXIT

EFFECTIVE_MODEL=$MODEL
if [[ "$STAGE_MODEL" == true ]]; then
  EFFECTIVE_MODEL=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/model
  mkdir -p "$EFFECTIVE_MODEL"
  cp -a "$MODEL"/. "$EFFECTIVE_MODEL"/
elif [[ "$STAGE_MODEL" != false ]]; then
  echo "STAGE_MODEL must be true or false" >&2
  exit 2
fi

echo "[crp1-eval] node=${SLURMD_NODENAME:-?} code=$CODE_ROOT commit=$SOURCE_COMMIT arm=$ARM variant=$VARIANT ablation=$ABLATION"
nvidia-smi -L
"$PYTHON" -m py_compile \
  "$CODE_ROOT/train/diverge_crp1_workspace.py" \
  "$CODE_ROOT/train/diverge_crp1_data.py" \
  "$CODE_ROOT/train/diverge_crp1_product.py" \
  "$CODE_ROOT/train/eval_diverge_crp1.py"

CHECKPOINT_ARGS=()
if [[ "$ARM" != plain ]]; then
  CHECKPOINT=${CHECKPOINT:?trained arm requires CHECKPOINT}
  CHECKPOINT_SHA256=${CHECKPOINT_SHA256:?trained arm requires CHECKPOINT_SHA256}
  test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$CHECKPOINT_SHA256"
  CHECKPOINT_ARGS+=(--checkpoint "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA256")
fi

"$PYTHON" "$CODE_ROOT/train/eval_diverge_crp1.py" \
  --model-root "$EFFECTIVE_MODEL" \
  --model-revision a07cc9a04f16550a088caea529712d1d335b0ac1 \
  --model-loader causal \
  --source-checkpoint "$SOURCE_CHECKPOINT" \
  --source-checkpoint-sha256 34c82454e0c53609bc1ac6a9f127437080e431f147e28ae63b4080c413d9a82e \
  --data "$DATA" --data-sha256 "$DATA_SHA256" \
  --output "$OUTPUT" --arm "$ARM" --ablation "$ABLATION" \
  --variant "$VARIANT" --count "$COUNT" --batch-size "$BATCH_SIZE" \
  --max-sequence-length 4096 --workspace-slots 6 --max-trace-steps 12 \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --subset-seed 47 --generation-seed 2026080604 \
  "${CHECKPOINT_ARGS[@]}"

sha256sum "$OUTPUT"
echo "[crp1-eval] completed arm=$ARM variant=$VARIANT ablation=$ABLATION"
