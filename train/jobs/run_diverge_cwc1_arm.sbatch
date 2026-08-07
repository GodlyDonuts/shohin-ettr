#!/bin/bash
#SBATCH -J diverge-cwc1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH --exclude=evc33,evc43
#SBATCH -o /lustre/fs1/home/%u/shohin/logs/diverge_cwc1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CWC1 runtime root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set frozen CWC1 source commit}
ARM=${ARM:?set involution, duplicate, or augmentation}
DATA=${DATA:?set immutable CWC1 training data}
DATA_SHA256=${DATA_SHA256:?set CWC1 training data hash}
OUTPUT=${OUTPUT:?set isolated CWC1 model output}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

case "$ARM" in
  involution|duplicate|augmentation) ;;
  *) echo "invalid CWC1 arm=$ARM" >&2; exit 2 ;;
esac
export PYTHONPATH="$CODE_ROOT/train"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs"

(cd "$CODE_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$DATA" | cut -d' ' -f1)" = "$DATA_SHA256"
test ! -e "$OUTPUT"
echo "[diverge-cwc1] node=${SLURMD_NODENAME:-?} arm=$ARM commit=$SOURCE_COMMIT"
nvidia-smi -L
"$PYTHON" -m py_compile \
  "$CODE_ROOT/train/diverge_cwc1_data.py" \
  "$CODE_ROOT/train/diverge_cwc1_runtime.py" \
  "$CODE_ROOT/train/train_diverge_cwc1.py"
"$PYTHON" "$CODE_ROOT/train/train_diverge_cwc1.py" \
  --data "$DATA" --data-sha256 "$DATA_SHA256" \
  --output "$OUTPUT" --arm "$ARM" \
  --updates 1000 --batch-size 128 --evaluation-batch-size 256 \
  --learning-rate 0.003 --seed 2026080731
chmod -R a-w "$OUTPUT"
echo "[diverge-cwc1] completed arm=$ARM"
