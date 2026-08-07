#!/bin/bash
#SBATCH -J npl2-confirm-seed
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH --exclude=evc33,evc43
#SBATCH -o /lustre/fs1/home/%u/shohin/logs/diverge_npl2_confirm_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable NPL2 confirmation code root}
DEV_RESULT=${DEV_RESULT:?set passing NPL2 development result}
DEV_RESULT_SHA256=${DEV_RESULT_SHA256:?set NPL2 development result hash}
OUTPUT=${OUTPUT:?set isolated NPL2 seed output}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs"
(cd "$CODE_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$DEV_RESULT" | cut -d' ' -f1)" = "$DEV_RESULT_SHA256"
test "$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$DEV_RESULT")" = pass
test ! -e "$OUTPUT"
nvidia-smi -L
"$PYTHON" "$CODE_ROOT/train/test_diverge_npl2_runtime.py"
"$PYTHON" "$CODE_ROOT/train/test_eval_diverge_npl2_development.py"
"$PYTHON" "$CODE_ROOT/train/eval_diverge_npl2_development.py" \
  --source-commit "${SOURCE_COMMIT:?set confirmation source commit}" \
  --eni1-result "${ENI1_RESULT:?}" --eni1-result-sha256 "${ENI1_RESULT_SHA256:?}" \
  --eic-checkpoint "${EIC_CHECKPOINT:?}" --eic-checkpoint-sha256 "${EIC_CHECKPOINT_SHA256:?}" \
  --base "${PARENT:?}" --base-sha256 "${PARENT_SHA256:?}" \
  --tokenizer "${TOKENIZER:?}" --tokenizer-sha256 "${TOKENIZER_SHA256:?}" \
  --sti-checkpoint "${STI_CHECKPOINT:?}" --sti-checkpoint-sha256 "${STI_CHECKPOINT_SHA256:?}" \
  --public-data "${PUBLIC_DATA:?}" --public-data-sha256 "${PUBLIC_DATA_SHA256:?}" \
  --assessor-data "${ASSESSOR_DATA:?}" --assessor-data-sha256 "${ASSESSOR_DATA_SHA256:?}" \
  --output "$OUTPUT" --batch-size 512 --evaluation-split confirmation
test "$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$OUTPUT")" = complete
