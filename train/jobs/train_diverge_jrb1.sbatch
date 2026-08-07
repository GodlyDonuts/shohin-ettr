#!/bin/bash
#SBATCH -J jrb1-train
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/%u/shohin/logs/diverge_jrb1_train_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable JRB1 code root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set frozen JRB1 source commit}
TRAINING_DATA=${TRAINING_DATA:?set immutable JRB1 training data}
TRAINING_SHA256=${TRAINING_SHA256:?set JRB1 training hash}
ARM=${ARM:?set treatment or shuffled_table}
OUTPUT=${OUTPUT:?set isolated JRB1 arm output}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs" "$(dirname "$OUTPUT")"
(cd "$CODE_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$TRAINING_DATA" | cut -d' ' -f1)" = "$TRAINING_SHA256"
test ! -e "$OUTPUT"
trap 'test ! -e "$OUTPUT" || chmod -R a-w "$OUTPUT"' EXIT
nvidia-smi -L
"$PYTHON" "$CODE_ROOT/train/test_diverge_jrb1.py"
"$PYTHON" "$CODE_ROOT/train/train_diverge_jrb1.py" \
  --data "$TRAINING_DATA" --data-sha256 "$TRAINING_SHA256" \
  --output "$OUTPUT" --source-commit "$SOURCE_COMMIT" --arm "$ARM" \
  --updates 1000 --batch-size 128 --learning-rate 0.001
echo "[diverge-jrb1-train] completed arm=$ARM"
