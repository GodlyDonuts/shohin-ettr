#!/bin/bash
#SBATCH -J product-candidate-verifier
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/product_candidate_verifier_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
RUNTIME=${RUNTIME:?set immutable runtime}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL_ROOT=${MODEL_ROOT:?set pinned model root}
MODEL_REVISION=${MODEL_REVISION:?set pinned model revision}
ADAPTER_CHECKPOINT=${ADAPTER_CHECKPOINT:?set adapter checkpoint}
CANDIDATES=${CANDIDATES:?set candidate source}
OUTPUT=${OUTPUT:?set scored candidate output}
REPORT=${REPORT:?set report output}
SKIP=${SKIP:-0}
COUNT=${COUNT:?set identity count}
MAX_SEQUENCE_LENGTH=${MAX_SEQUENCE_LENGTH:-4096}
STAGE_MODEL=${STAGE_MODEL:-true}

test -x "$PYTHON"
for path in "$RUNTIME/train/hf_product_candidate_verifier.py" "$ADAPTER_CHECKPOINT" "$CANDIDATES"; do
  [ -f "$path" ] || { echo "[candidate-verifier] missing input: $path" >&2; exit 2; }
done
for path in "$OUTPUT" "$REPORT"; do
  [ ! -e "$path" ] || { echo "[candidate-verifier] refusing existing output: $path" >&2; exit 3; }
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
elif [[ "$STAGE_MODEL" != "false" ]]; then
  echo "STAGE_MODEL must be true or false" >&2
  exit 4
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
cd "$RUNTIME/train"
"$PYTHON" hf_product_candidate_verifier.py \
  --model-root "$EFFECTIVE_MODEL_ROOT" \
  --model-source-root "$SOURCE_MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --adapter-checkpoint "$ADAPTER_CHECKPOINT" \
  --candidates "$CANDIDATES" \
  --output "$OUTPUT" \
  --report "$REPORT" \
  --skip "$SKIP" \
  --count "$COUNT" \
  --max-sequence-length "$MAX_SEQUENCE_LENGTH"
