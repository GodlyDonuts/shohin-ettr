#!/bin/bash
#SBATCH -J cwc1-confirm
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH --exclude=evc33,evc43
#SBATCH -o /lustre/fs1/home/%u/shohin/logs/diverge_cwc1_confirm_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CWC1 runtime root}
DATA=${DATA:?set sealed CWC1 confirmation board}
DATA_SHA256=${DATA_SHA256:?set CWC1 confirmation board hash}
INVOLUTION=${INVOLUTION:?set qualified involution model root}
DEVELOPMENT_RESULT=${DEVELOPMENT_RESULT:?set passing CWC1 development result}
OUTPUT=${OUTPUT:?set isolated CWC1 confirmation result}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

export PYTHONPATH="$CODE_ROOT/train"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs"
(cd "$CODE_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$DATA" | cut -d' ' -f1)" = "$DATA_SHA256"
test ! -e "$OUTPUT"
DEVELOPMENT_SHA256=$(sha256sum "$DEVELOPMENT_RESULT" | cut -d' ' -f1)
test "$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["all_pass"])' "$DEVELOPMENT_RESULT")" = True
nvidia-smi -L
"$PYTHON" "$CODE_ROOT/train/eval_diverge_cwc1.py" \
  --board "$DATA" --board-sha256 "$DATA_SHA256" --split confirmation \
  --involution-root "$INVOLUTION" \
  --development-result "$DEVELOPMENT_RESULT" \
  --development-result-sha256 "$DEVELOPMENT_SHA256" \
  --output "$OUTPUT" --batch-size 256
chmod a-w "$OUTPUT"
test "$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["all_pass"])' "$OUTPUT")" = True
