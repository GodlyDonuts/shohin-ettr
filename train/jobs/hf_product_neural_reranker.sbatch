#!/bin/bash
#SBATCH -J product-neural-reranker
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc32,evc33,evc40,evc43,evc44
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/product_neural_reranker_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
RUNTIME=${RUNTIME:?set immutable runtime}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODE=${MODE:-fit}
FEATURES_GLOB=${FEATURES_GLOB:?set feature glob}
MODEL=${MODEL:?set model output/input}
OUTPUT=${OUTPUT:?set output path}
SHAPE_MODEL=${SHAPE_MODEL:-}

test -x "$PYTHON"
test -f "$RUNTIME/pipeline/product_neural_candidate_reranker.py"
mapfile -t FEATURES < <(compgen -G "$FEATURES_GLOB" | sort)
[ "${#FEATURES[@]}" -gt 0 ] || { echo "[neural-reranker] no feature shards" >&2; exit 2; }
FEATURE_ARGS=()
for path in "${FEATURES[@]}"; do FEATURE_ARGS+=(--features "$path"); done

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
cd "$RUNTIME/pipeline"
if [[ "$MODE" == "fit" ]]; then
  test -n "$SHAPE_MODEL" && test -f "$SHAPE_MODEL"
  [ ! -e "$MODEL" ] || { echo "[neural-reranker] refusing existing model" >&2; exit 3; }
  "$PYTHON" product_neural_candidate_reranker.py fit \
    "${FEATURE_ARGS[@]}" --shape-model "$SHAPE_MODEL" --output "$MODEL"
elif [[ "$MODE" == "select" ]]; then
  test -f "$MODEL"
  [ ! -e "$OUTPUT" ] || { echo "[neural-reranker] refusing existing output" >&2; exit 3; }
  "$PYTHON" product_neural_candidate_reranker.py select \
    "${FEATURE_ARGS[@]}" --model "$MODEL" --output "$OUTPUT"
else
  echo "MODE must be fit or select" >&2
  exit 4
fi
