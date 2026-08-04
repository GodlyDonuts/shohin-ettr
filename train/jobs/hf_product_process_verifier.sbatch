#!/bin/bash
#SBATCH -J product-process-verifier
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc32,evc33,evc40,evc43,evc44
#SBATCH -c 4
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/product_process_verifier_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
RUNTIME=${RUNTIME:?set immutable runtime}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL_ROOT=${MODEL_ROOT:?set pinned model root}
MODEL_REVISION=${MODEL_REVISION:?set pinned model revision}
ADAPTER_CHECKPOINT=${ADAPTER_CHECKPOINT:?set protected generator adapter}
CANDIDATES=${CANDIDATES:?set corrected candidate source}
SHAPE_MODEL=${SHAPE_MODEL:?set frozen shape reranker}
OUTPUT=${OUTPUT:?set isolated verifier output}
UPDATES=${UPDATES:-300}
MIN_UPDATES=${MIN_UPDATES:-100}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-8}
PAIRS_PER_PROMPT=${PAIRS_PER_PROMPT:-2}
TRAIN_SCOPE=${TRAIN_SCOPE:-leader}
BACKBONE_LR=${BACKBONE_LR:-2e-6}
HEAD_LR=${HEAD_LR:-2e-4}
SEED=${SEED:-20260809}
STAGE_MODEL=${STAGE_MODEL:-true}

test -x "$PYTHON"
for path in \
  "$RUNTIME/train/hf_product_process_verifier.py" \
  "$RUNTIME/train/hf_product_reasoning_train.py" \
  "$RUNTIME/train/hf_product_reasoning_eval.py" \
  "$RUNTIME/pipeline/product_candidate_reranker.py" \
  "$ADAPTER_CHECKPOINT" "$CANDIDATES" "$SHAPE_MODEL"; do
  [ -f "$path" ] || { echo "[process-verifier] missing input: $path" >&2; exit 2; }
done
[ ! -e "$OUTPUT" ] || { echo "[process-verifier] refusing output: $OUTPUT" >&2; exit 3; }

SOURCE_MODEL_ROOT=$MODEL_ROOT
EFFECTIVE_MODEL_ROOT=$MODEL_ROOT
if [[ "$STAGE_MODEL" == "true" ]]; then
  LOCAL_PARENT=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
  EFFECTIVE_MODEL_ROOT=$LOCAL_PARENT/model
  mkdir -p "$EFFECTIVE_MODEL_ROOT"
  cp -a "$SOURCE_MODEL_ROOT"/. "$EFFECTIVE_MODEL_ROOT"/
  test -f "$EFFECTIVE_MODEL_ROOT/config.json"
elif [[ "$STAGE_MODEL" != "false" ]]; then
  echo "STAGE_MODEL must be true or false" >&2
  exit 4
fi

mkdir -p logs "$(dirname "$OUTPUT")"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
cd "$RUNTIME/train"
"$PYTHON" hf_product_process_verifier.py train \
  --model-root "$EFFECTIVE_MODEL_ROOT" \
  --model-source-root "$SOURCE_MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --adapter-checkpoint "$ADAPTER_CHECKPOINT" \
  --candidates "$CANDIDATES" \
  --shape-model "$SHAPE_MODEL" \
  --output "$OUTPUT" \
  --updates "$UPDATES" \
  --min-updates "$MIN_UPDATES" \
  --gradient-accumulation "$GRADIENT_ACCUMULATION" \
  --pairs-per-prompt "$PAIRS_PER_PROMPT" \
  --train-scope "$TRAIN_SCOPE" \
  --backbone-learning-rate "$BACKBONE_LR" \
  --head-learning-rate "$HEAD_LR" \
  --seed "$SEED"
