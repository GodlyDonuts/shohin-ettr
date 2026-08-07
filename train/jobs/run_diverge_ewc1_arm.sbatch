#!/bin/bash
#SBATCH -J diverge-ewc1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH --exclude=evc33,evc43
#SBATCH -o /lustre/fs1/home/%u/shohin/logs/diverge_ewc1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable EWC1 runtime root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set EWC1 source commit}
MODE=${MODE:?set equivariant or absolute}
DATA=${DATA:?set immutable EWC1 training data}
DATA_SHA256=${DATA_SHA256:?set EWC1 training data hash}
OUTPUT=${OUTPUT:?set isolated EWC1 model output}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

case "$MODE" in
  equivariant|absolute) ;;
  *) echo "invalid EWC1 mode=$MODE" >&2; exit 2 ;;
esac
export PYTHONPATH="$CODE_ROOT/train"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs"

(cd "$CODE_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$DATA" | cut -d' ' -f1)" = "$DATA_SHA256"
test ! -e "$OUTPUT"
echo "[diverge-ewc1] node=${SLURMD_NODENAME:-?} mode=$MODE commit=$SOURCE_COMMIT"
nvidia-smi -L
"$PYTHON" -m py_compile \
  "$CODE_ROOT/train/diverge_ewc1_data.py" \
  "$CODE_ROOT/train/diverge_ewc1_runtime.py" \
  "$CODE_ROOT/train/train_diverge_ewc1.py"
"$PYTHON" "$CODE_ROOT/train/train_diverge_ewc1.py" \
  --data "$DATA" --data-sha256 "$DATA_SHA256" \
  --output "$OUTPUT" --mode "$MODE" \
  --updates 1000 --batch-size 256 --evaluation-batch-size 512 \
  --learning-rate 0.003 --seed 2026080721
chmod -R a-w "$OUTPUT"
echo "[diverge-ewc1] completed mode=$MODE"
