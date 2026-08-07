#!/bin/bash
#SBATCH -J eic1-confirm
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 00:30:00
#SBATCH --exclude=evc33,evc43
#SBATCH -o /lustre/fs1/home/%u/shohin/logs/eic1-confirm-%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
OVERLAY_ROOT=${OVERLAY_ROOT:?set immutable EIC1 confirmation root}
RESULT_ROOT=${RESULT_ROOT:?set immutable selected EIC1 treatment root}
ASSESSMENT=${ASSESSMENT:?set passing EIC1 assessment}
ASSESSMENT_SHA256=${ASSESSMENT_SHA256:?set assessment SHA-256}
BOARD=${BOARD:?set sealed EIC1 board}
BOARD_SHA256=${BOARD_SHA256:?set sealed EIC1 board SHA-256}
BOARD_REPORT=${BOARD_REPORT:?set sealed EIC1 board report}
BOARD_REPORT_SHA256=${BOARD_REPORT_SHA256:?set board report SHA-256}
OUTPUT=${OUTPUT:?set one EIC1 confirmation output}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
PARENT=$BASE/train/flagship_out/ckpt_0300000.pt
PARENT_SHA256=211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6
TOKENIZER=$BASE/artifacts/shohin-tok-32k.json
TOKENIZER_SHA256=87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4
CHECKPOINT=$RESULT_ROOT/model/checkpoint.pt

export PYTHONPATH="$OVERLAY_ROOT/train:$OVERLAY_ROOT/pipeline"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs"
(cd "$OVERLAY_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
"$PYTHON" "$OVERLAY_ROOT/train/test_authorize_diverge_eic1_confirmation.py"
"$PYTHON" "$OVERLAY_ROOT/train/authorize_diverge_eic1_confirmation.py" \
  --assessment "$ASSESSMENT" --assessment-sha256 "$ASSESSMENT_SHA256" \
  --board "$BOARD" --board-sha256 "$BOARD_SHA256" \
  --board-report "$BOARD_REPORT" --board-report-sha256 "$BOARD_REPORT_SHA256"

CHECKPOINT_SHA256=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
test ! -e "$OUTPUT"
nvidia-smi -L
"$PYTHON" "$OVERLAY_ROOT/train/eval_diverge_eic1.py" \
  --checkpoint "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --base "$PARENT" --base-sha256 "$PARENT_SHA256" \
  --tokenizer "$TOKENIZER" --tokenizer-sha256 "$TOKENIZER_SHA256" \
  --data "$BOARD" --data-sha256 "$BOARD_SHA256" --board-type confirmation \
  --output "$OUTPUT" --batch-size 32 --device cuda
