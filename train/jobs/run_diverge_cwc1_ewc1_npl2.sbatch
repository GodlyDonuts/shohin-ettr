#!/bin/bash
#SBATCH -J cwc1-npl2
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH --exclude=evc33,evc43
#SBATCH -o /lustre/fs1/home/%u/shohin/logs/diverge_cwc1_ewc1_npl2_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
BASE_CODE_ROOT=${BASE_CODE_ROOT:?set immutable confirmed NPL2 runtime}
OVERLAY_ROOT=${OVERLAY_ROOT:?set immutable CWC1/EWC1/NPL2 overlay}
CWC_ROOT=${CWC_ROOT:?set immutable confirmed CWC1 involution root}
CWC_RESULT=${CWC_RESULT:?set immutable CWC1 confirmation}
EWC_CHECKPOINT=${EWC_CHECKPOINT:?set frozen EWC1 structural checkpoint}
EWC_RESULT=${EWC_RESULT:?set frozen EWC1 development result}
ENI1_RESULT=${ENI1_RESULT:?set passing ENI1 receipt}
EIC_CHECKPOINT=${EIC_CHECKPOINT:?set confirmed EIC1 checkpoint}
STI_CHECKPOINT=${STI_CHECKPOINT:?set protected STI1 checkpoint}
PUBLIC_DATA=${PUBLIC_DATA:?set immutable NPL2 public data}
PUBLIC_DATA_SHA256=${PUBLIC_DATA_SHA256:?set NPL2 public hash}
ASSESSOR_DATA=${ASSESSOR_DATA:?set immutable NPL2 assessor data}
ASSESSOR_DATA_SHA256=${ASSESSOR_DATA_SHA256:?set NPL2 assessor hash}
WRAPPER_DATA=${WRAPPER_DATA:?set immutable CWC1 wrapper data}
WRAPPER_DATA_SHA256=${WRAPPER_DATA_SHA256:?set CWC1 wrapper hash}
OUTPUT=${OUTPUT:?set isolated integration output}
EVALUATION_SPLIT=${EVALUATION_SPLIT:-development}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
PARENT=$BASE/train/flagship_out/ckpt_0300000.pt
PARENT_SHA256=211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6
TOKENIZER=$BASE/artifacts/shohin-tok-32k.json
TOKENIZER_SHA256=87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4
CWC_CHECKPOINT_SHA256=cae4d89611c8b4acd6182d337b3d5955dd95bbd6b375be227c3ae9ee8617ab1a
CWC_TRAINING_REPORT_SHA256=62998b8b5244b68aa2ac1b55170e92c6afe1978e8ceb476ce3944bd5faf85f59
CWC_RESULT_SHA256=f42802ceeb6bae47b3c6f66d1c08be3808423a3ed91c03f59d05f3979ecf5a9d
EWC_CHECKPOINT_SHA256=0816ed1c65d51951cf11625f505ea7886a176c008cc42ac3ac821f369788d1ed
EWC_RESULT_SHA256=2e173969609a4d50f7fb169b18d56611033a28a03c7c71d618c873a9ce1e5760
ENI1_RESULT_SHA256=503ec03578c39db2ef3e678842ec84f080f56662aae5c6debad8be4cca34eca5
EIC_CHECKPOINT_SHA256=cd86f7335b9b7df078acc4b17c4dbe6b8bf944daee40f5397b452a0b5fae8849
STI_CHECKPOINT_SHA256=05c7b6c7ee68e14133231d572b8c66aced5e3a72a6ba25827498560f723336ae

case "$EVALUATION_SPLIT" in
  development) ;;
  confirmation)
    test "$(sha256sum "${DEV_RESULT:?set passing integration development result}" | cut -d' ' -f1)" = "${DEV_RESULT_SHA256:?set development result hash}"
    test "$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$DEV_RESULT")" = pass
    ;;
  *) echo "invalid CWC1/EWC1/NPL2 evaluation split" >&2; exit 2 ;;
esac
export PYTHONPATH="$OVERLAY_ROOT/train:$BASE_CODE_ROOT/train:$BASE_CODE_ROOT/pipeline"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs"
(cd "$BASE_CODE_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
(cd "$OVERLAY_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
test ! -e "$OUTPUT"
nvidia-smi -L
"$PYTHON" "$OVERLAY_ROOT/train/test_diverge_cwc1_npl2.py"
"$PYTHON" "$OVERLAY_ROOT/train/eval_diverge_cwc1_ewc1_npl2.py" \
  --cwc-root "$CWC_ROOT" \
  --cwc-checkpoint-sha256 "$CWC_CHECKPOINT_SHA256" \
  --cwc-training-report-sha256 "$CWC_TRAINING_REPORT_SHA256" \
  --cwc-result "$CWC_RESULT" --cwc-result-sha256 "$CWC_RESULT_SHA256" \
  --cwc-wrapper-data "$WRAPPER_DATA" \
  --cwc-wrapper-data-sha256 "$WRAPPER_DATA_SHA256" \
  --ewc-checkpoint "$EWC_CHECKPOINT" \
  --ewc-checkpoint-sha256 "$EWC_CHECKPOINT_SHA256" \
  --ewc-result "$EWC_RESULT" --ewc-result-sha256 "$EWC_RESULT_SHA256" \
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
