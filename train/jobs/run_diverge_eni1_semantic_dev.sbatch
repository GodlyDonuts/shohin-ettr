#!/bin/bash
#SBATCH -J eni1-sem-dev
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH --exclude=evc33,evc43
#SBATCH -o /lustre/fs1/home/%u/shohin/logs/diverge_eni1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable ENI1 code root}
EIC_CHECKPOINT=${EIC_CHECKPOINT:?set confirmed EIC1 checkpoint}
EIC_CHECKPOINT_SHA256=${EIC_CHECKPOINT_SHA256:?set EIC1 checkpoint hash}
STI_CHECKPOINT=${STI_CHECKPOINT:?set protected STI1 checkpoint}
STI_CHECKPOINT_SHA256=${STI_CHECKPOINT_SHA256:?set STI1 checkpoint hash}
PUBLIC_DATA=${PUBLIC_DATA:?set immutable NPL1 public development data}
PUBLIC_DATA_SHA256=${PUBLIC_DATA_SHA256:?set NPL1 public hash}
ASSESSOR_DATA=${ASSESSOR_DATA:?set immutable NPL1 assessor development data}
ASSESSOR_DATA_SHA256=${ASSESSOR_DATA_SHA256:?set NPL1 assessor hash}
OUTPUT=${OUTPUT:?set isolated ENI1 output}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
PARENT=$BASE/train/flagship_out/ckpt_0300000.pt
PARENT_SHA256=211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6
TOKENIZER=$BASE/artifacts/shohin-tok-32k.json
TOKENIZER_SHA256=87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4

export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs"

(cd "$CODE_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
test ! -e "$OUTPUT"
nvidia-smi -L
"$PYTHON" "$CODE_ROOT/train/test_eval_diverge_eni1_semantics.py"
"$PYTHON" "$CODE_ROOT/train/eval_diverge_eni1_semantics.py" \
  --eic-checkpoint "$EIC_CHECKPOINT" \
  --eic-checkpoint-sha256 "$EIC_CHECKPOINT_SHA256" \
  --base "$PARENT" --base-sha256 "$PARENT_SHA256" \
  --tokenizer "$TOKENIZER" --tokenizer-sha256 "$TOKENIZER_SHA256" \
  --sti-checkpoint "$STI_CHECKPOINT" \
  --sti-checkpoint-sha256 "$STI_CHECKPOINT_SHA256" \
  --public-data "$PUBLIC_DATA" --public-data-sha256 "$PUBLIC_DATA_SHA256" \
  --assessor-data "$ASSESSOR_DATA" --assessor-data-sha256 "$ASSESSOR_DATA_SHA256" \
  --output "$OUTPUT" --batch-size 32 --device cuda

STATUS=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$OUTPUT")
test "$STATUS" = pass

