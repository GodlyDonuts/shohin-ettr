#!/bin/bash
#SBATCH -J shohin-product-rollouts
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/hf_product_rollouts_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
RUNTIME=${RUNTIME:?set immutable runtime}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL_ROOT=${MODEL_ROOT:?set pinned model root}
MODEL_REVISION=${MODEL_REVISION:?set pinned model revision}
ADAPTER_CHECKPOINT=${ADAPTER_CHECKPOINT:?set adapter checkpoint}
DATA=${DATA:?set fresh rollout bank}
CANDIDATES_OUTPUT=${CANDIDATES_OUTPUT:?set candidate output}
POSITIVES_OUTPUT=${POSITIVES_OUTPUT:?set positive output}
REPORT=${REPORT:?set report output}
SKIP=${SKIP:-0}
COUNT=${COUNT:-128}
SAMPLES=${SAMPLES:-4}
PROMPT_BATCH_SIZE=${PROMPT_BATCH_SIZE:-1}
SEED=${SEED:-20260803}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1536}
STAGE_MODEL=${STAGE_MODEL:-true}

test -x "$PYTHON"
for path in "$RUNTIME/train/hf_product_reasoning_rollouts.py" "$ADAPTER_CHECKPOINT" "$DATA"; do
  [ -f "$path" ] || { echo "[product-rollout] missing input: $path" >&2; exit 2; }
done
for path in "$CANDIDATES_OUTPUT" "$POSITIVES_OUTPUT" "$REPORT"; do
  [ ! -e "$path" ] || { echo "[product-rollout] refusing existing output: $path" >&2; exit 3; }
  mkdir -p "$(dirname "$path")"
done

SOURCE_MODEL_ROOT=$MODEL_ROOT
EFFECTIVE_MODEL_ROOT=$MODEL_ROOT
if [[ "$STAGE_MODEL" == "true" ]]; then
  LOCAL_PARENT=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
  EFFECTIVE_MODEL_ROOT=$LOCAL_PARENT/model
  mkdir -p "$EFFECTIVE_MODEL_ROOT"
  cp -a "$SOURCE_MODEL_ROOT"/. "$EFFECTIVE_MODEL_ROOT"/
  test -f "$EFFECTIVE_MODEL_ROOT/config.json"
  test -f "$EFFECTIVE_MODEL_ROOT/model.safetensors.index.json"
elif [[ "$STAGE_MODEL" != "false" ]]; then
  echo "STAGE_MODEL must be true or false" >&2
  exit 4
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
cd "$RUNTIME/train"
"$PYTHON" hf_product_reasoning_rollouts.py \
  --model-root "$EFFECTIVE_MODEL_ROOT" \
  --model-source-root "$SOURCE_MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --adapter-checkpoint "$ADAPTER_CHECKPOINT" \
  --data "$DATA" \
  --candidates-output "$CANDIDATES_OUTPUT" \
  --positives-output "$POSITIVES_OUTPUT" \
  --report "$REPORT" \
  --skip "$SKIP" \
  --count "$COUNT" \
  --samples "$SAMPLES" \
  --prompt-batch-size "$PROMPT_BATCH_SIZE" \
  --seed "$SEED" \
  --max-new-tokens "$MAX_NEW_TOKENS"
