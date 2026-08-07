#!/bin/bash
#SBATCH -J ewc1-npl2
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH --exclude=evc33,evc43
#SBATCH -o /lustre/fs1/home/%u/shohin/logs/diverge_ewc1_npl2_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable EWC1/NPL2 runtime root}
EWC_CHECKPOINT=${EWC_CHECKPOINT:?set confirmed EWC1 checkpoint}
EWC_RESULT=${EWC_RESULT:?set passing EWC1 confirmation}
ENI1_RESULT=${ENI1_RESULT:?set passing ENI1 receipt}
EIC_CHECKPOINT=${EIC_CHECKPOINT:?set confirmed EIC1 checkpoint}
STI_CHECKPOINT=${STI_CHECKPOINT:?set protected STI1 checkpoint}
PUBLIC_DATA=${PUBLIC_DATA:?set immutable NPL public data}
PUBLIC_DATA_SHA256=${PUBLIC_DATA_SHA256:?set NPL public hash}
ASSESSOR_DATA=${ASSESSOR_DATA:?set immutable NPL assessor data}
ASSESSOR_DATA_SHA256=${ASSESSOR_DATA_SHA256:?set NPL assessor hash}
OUTPUT=${OUTPUT:?set isolated EWC1/NPL2 result}
EVALUATION_SPLIT=${EVALUATION_SPLIT:-development}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
PARENT=$BASE/train/flagship_out/ckpt_0300000.pt
PARENT_SHA256=211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6
TOKENIZER=$BASE/artifacts/shohin-tok-32k.json
TOKENIZER_SHA256=87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4
ENI1_RESULT_SHA256=503ec03578c39db2ef3e678842ec84f080f56662aae5c6debad8be4cca34eca5
EIC_CHECKPOINT_SHA256=cd86f7335b9b7df078acc4b17c4dbe6b8bf944daee40f5397b452a0b5fae8849
STI_CHECKPOINT_SHA256=05c7b6c7ee68e14133231d572b8c66aced5e3a72a6ba25827498560f723336ae

case "$EVALUATION_SPLIT" in
  development|confirmation) ;;
  *) echo "invalid EWC1/NPL2 evaluation split" >&2; exit 2 ;;
esac
export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs"
(cd "$CODE_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
test ! -e "$OUTPUT"
EWC_CHECKPOINT_SHA256=$(sha256sum "$EWC_CHECKPOINT" | cut -d' ' -f1)
EWC_RESULT_SHA256=$(sha256sum "$EWC_RESULT" | cut -d' ' -f1)
nvidia-smi -L
"$PYTHON" -m py_compile \
  "$CODE_ROOT/train/eval_diverge_ewc1_npl2.py" \
  "$CODE_ROOT/train/eval_diverge_npl2_development.py"
"$PYTHON" "$CODE_ROOT/train/eval_diverge_ewc1_npl2.py" \
  --ewc-checkpoint "$EWC_CHECKPOINT" \
  --ewc-checkpoint-sha256 "$EWC_CHECKPOINT_SHA256" \
  --ewc-result "$EWC_RESULT" --ewc-result-sha256 "$EWC_RESULT_SHA256" \
  --ewc-batch-size 512 \
  --source-commit "${SOURCE_COMMIT:?set integration source commit}" \
  --eni1-result "$ENI1_RESULT" --eni1-result-sha256 "$ENI1_RESULT_SHA256" \
  --eic-checkpoint "$EIC_CHECKPOINT" \
  --eic-checkpoint-sha256 "$EIC_CHECKPOINT_SHA256" \
  --base "$PARENT" --base-sha256 "$PARENT_SHA256" \
  --tokenizer "$TOKENIZER" --tokenizer-sha256 "$TOKENIZER_SHA256" \
  --sti-checkpoint "$STI_CHECKPOINT" \
  --sti-checkpoint-sha256 "$STI_CHECKPOINT_SHA256" \
  --public-data "$PUBLIC_DATA" --public-data-sha256 "$PUBLIC_DATA_SHA256" \
  --assessor-data "$ASSESSOR_DATA" --assessor-data-sha256 "$ASSESSOR_DATA_SHA256" \
  --output "$OUTPUT" --batch-size 512 --evaluation-split "$EVALUATION_SPLIT"
if [[ "$EVALUATION_SPLIT" == development ]]; then
  test "$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$OUTPUT")" = pass
else
  test "$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$OUTPUT")" = complete
fi
