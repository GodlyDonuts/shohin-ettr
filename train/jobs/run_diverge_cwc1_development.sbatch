#!/bin/bash
#SBATCH -J cwc1-dev
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:30:00
#SBATCH --exclude=evc33,evc43
#SBATCH -o /lustre/fs1/home/%u/shohin/logs/diverge_cwc1_dev_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CWC1 runtime root}
DATA=${DATA:?set immutable CWC1 development board}
DATA_SHA256=${DATA_SHA256:?set CWC1 development board hash}
INVOLUTION=${INVOLUTION:?set involution model root}
DUPLICATE=${DUPLICATE:?set duplicate model root}
AUGMENTATION=${AUGMENTATION:?set augmentation model root}
OUTPUT=${OUTPUT:?set isolated CWC1 development result}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

export PYTHONPATH="$CODE_ROOT/train"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs"
(cd "$CODE_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$DATA" | cut -d' ' -f1)" = "$DATA_SHA256"
test ! -e "$OUTPUT"
nvidia-smi -L
"$PYTHON" -m py_compile "$CODE_ROOT/train/eval_diverge_cwc1.py"
"$PYTHON" "$CODE_ROOT/train/eval_diverge_cwc1.py" \
  --board "$DATA" --board-sha256 "$DATA_SHA256" --split development \
  --involution-root "$INVOLUTION" --duplicate-root "$DUPLICATE" \
  --augmentation-root "$AUGMENTATION" --output "$OUTPUT" --batch-size 256
chmod a-w "$OUTPUT"
test "$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["all_pass"])' "$OUTPUT")" = True
