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
PY=$BASE/miniforge3/bin/python
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
SEED=${SEED:-20260803}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1536}

for path in "$RUNTIME/train/hf_product_reasoning_rollouts.py" "$ADAPTER_CHECKPOINT" "$DATA"; do
  [ -f "$path" ] || { echo "[product-rollout] missing input: $path" >&2; exit 2; }
done
for path in "$CANDIDATES_OUTPUT" "$POSITIVES_OUTPUT" "$REPORT"; do
  [ ! -e "$path" ] || { echo "[product-rollout] refusing existing output: $path" >&2; exit 3; }
  mkdir -p "$(dirname "$path")"
done

cd "$RUNTIME/train"
"$PY" hf_product_reasoning_rollouts.py \
  --model-root "$MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --adapter-checkpoint "$ADAPTER_CHECKPOINT" \
  --data "$DATA" \
  --candidates-output "$CANDIDATES_OUTPUT" \
  --positives-output "$POSITIVES_OUTPUT" \
  --report "$REPORT" \
  --skip "$SKIP" \
  --count "$COUNT" \
  --samples "$SAMPLES" \
  --seed "$SEED" \
  --max-new-tokens "$MAX_NEW_TOKENS"
