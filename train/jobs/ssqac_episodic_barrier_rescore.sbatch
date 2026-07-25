#!/bin/bash
#SBATCH -J ssqac-barrier
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH --exclude=evc26,evc31,evc33,evc43,evc44
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/ssqac_barrier_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=$BASE/miniforge3/bin/python3
RESULT_ROOT=$BASE/artifacts/r12/ssqac_episodic_anti_cycle/barrier

: "${PREPARATION:?PREPARATION must name the sealed training artifact}"
: "${MODEL:?MODEL must name one frozen episodic treatment model}"
: "${EVALUATION_SEED:?EVALUATION_SEED must be preregistered}"
EVALUATION_MATRICES=${EVALUATION_MATRICES:-512}
OUTPUT=$RESULT_ROOT/report_${SLURM_JOB_ID}.json

mkdir -p "$BASE/logs" "$RESULT_ROOT"
test -s "$PREPARATION"
test -s "$MODEL"
test ! -e "$OUTPUT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=$BASE:$BASE/train
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[ssqac-barrier] node=${SLURMD_NODENAME:-?} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[ssqac-barrier] model_sha256=$(sha256sum "$MODEL" | awk '{print $1}')"
nvidia-smi -L
"$PYTHON" "$BASE/train/ssqac_episodic_anti_cycle_controller.py" rescore \
  --preparation "$PREPARATION" \
  --model "$MODEL" \
  --evaluation-seed "$EVALUATION_SEED" \
  --evaluation-matrices "$EVALUATION_MATRICES" \
  --device cuda \
  --output "$OUTPUT"

test -s "$OUTPUT"
sha256sum "$OUTPUT"
