#!/bin/bash
#SBATCH -J ewc1-dev
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 01:00:00
#SBATCH --exclude=evc33,evc43
#SBATCH -o /lustre/fs1/home/%u/shohin/logs/diverge_ewc1_dev_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable EWC1 runtime root}
DATA=${DATA:?set immutable EWC1 development board}
DATA_SHA256=${DATA_SHA256:?set EWC1 development board hash}
TREATMENT=${TREATMENT:?set treatment model root}
CONTROL=${CONTROL:?set absolute-control model root}
OUTPUT=${OUTPUT:?set isolated EWC1 development result}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

export PYTHONPATH="$CODE_ROOT/train"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs"
(cd "$CODE_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$DATA" | cut -d' ' -f1)" = "$DATA_SHA256"
test ! -e "$OUTPUT"
nvidia-smi -L
"$PYTHON" -m py_compile "$CODE_ROOT/train/eval_diverge_ewc1.py"

TREATMENT_CKPT=$TREATMENT/checkpoint_0001000.pt
CONTROL_CKPT=$CONTROL/checkpoint_0001000.pt
TREATMENT_REPORT=$TREATMENT/report.json
CONTROL_REPORT=$CONTROL/report.json
TREATMENT_CKPT_SHA=$(sha256sum "$TREATMENT_CKPT" | cut -d' ' -f1)
CONTROL_CKPT_SHA=$(sha256sum "$CONTROL_CKPT" | cut -d' ' -f1)
TREATMENT_REPORT_SHA=$(sha256sum "$TREATMENT_REPORT" | cut -d' ' -f1)
CONTROL_REPORT_SHA=$(sha256sum "$CONTROL_REPORT" | cut -d' ' -f1)
"$PYTHON" "$CODE_ROOT/train/eval_diverge_ewc1.py" \
  --board "$DATA" --board-sha256 "$DATA_SHA256" \
  --checkpoint "$TREATMENT_CKPT" --checkpoint-sha256 "$TREATMENT_CKPT_SHA" \
  --control-checkpoint "$CONTROL_CKPT" --control-checkpoint-sha256 "$CONTROL_CKPT_SHA" \
  --training-report "$TREATMENT_REPORT" --training-report-sha256 "$TREATMENT_REPORT_SHA" \
  --control-training-report "$CONTROL_REPORT" \
  --control-training-report-sha256 "$CONTROL_REPORT_SHA" \
  --split development --output "$OUTPUT" --batch-size 512
chmod a-w "$OUTPUT"
test "$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["all_pass"])' "$OUTPUT")" = True
