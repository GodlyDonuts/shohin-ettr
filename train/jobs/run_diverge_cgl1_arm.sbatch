#!/bin/bash
#SBATCH -J diverge-cgl1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH --exclude=evc33,evc43
#SBATCH -o /lustre/fs1/home/%u/shohin/logs/diverge_cgl1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
BASE_CODE_ROOT=${BASE_CODE_ROOT:?set immutable qualified base code root}
OVERLAY_ROOT=${OVERLAY_ROOT:?set immutable CGL1 overlay root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set immutable CGL1 source commit}
BACKBONE=${BACKBONE:?set shohin or smollm2}
FLIP=${FLIP:-0}
OUT=${OUT:?set isolated CGL1 output root}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
PUBLIC_DATA=$BASE/artifacts/reasoning/diverge_cgl1/data_e4af112_r1/public.jsonl
PUBLIC_DATA_SHA256=bc438f793a3ced67a3b5493d70c14cbc39db4c20f3fe0fb50579af6b5f1daea9
SUPERVISOR_DATA=$BASE/artifacts/reasoning/diverge_cgl1/data_e4af112_r1/supervisor.jsonl
SUPERVISOR_DATA_SHA256=affa2cc36412f07f2816a00bbe2abfb06ee93be3b602c79b79b248f4ccf2552d
DEV_DATA=$BASE/artifacts/reasoning/diverge_ccr1/data_ec6d2f3_r1/confirmation_board.jsonl
DEV_DATA_SHA256=299237068f436ba33a68487b5300fcd724f8c98bd8bfe6b1916a4ebc7541ebf7

case "$BACKBONE" in
  shohin)
    PARENT=$BASE/train/flagship_out/ckpt_0300000.pt
    PARENT_SHA256=211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6
    TOKENIZER=$BASE/artifacts/shohin-tok-32k.json
    TOKENIZER_SHA256=87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4
    ;;
  smollm2)
    PARENT=$BASE/train/ettr_smollm2_control_parent_a2026072801_7881d8e/joint-model-final.pt
    PARENT_SHA256=8196f810a31e0abe7f3bf0eae0a37b103195f109b7a8e962c7b74b5710c98a02
    TOKENIZER=$BASE/artifacts/external/smollm2_135m_instruct_83212e1e/tokenizer.json
    TOKENIZER_SHA256=9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c
    ;;
  *) echo "invalid BACKBONE=$BACKBONE" >&2; exit 2 ;;
esac

EXTRA=()
if [ "$FLIP" = 1 ]; then
  EXTRA+=(--flip-outcomes)
elif [ "$FLIP" != 0 ]; then
  echo "FLIP must be 0 or 1" >&2
  exit 2
fi

MODEL_OUT=$OUT/model
DEV_OUT=$OUT/development.json
export PYTHONPATH="$OVERLAY_ROOT/train:$BASE_CODE_ROOT/train:$BASE_CODE_ROOT/pipeline"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs"
cd "$BASE_CODE_ROOT"

for path in "$PARENT" "$TOKENIZER" "$PUBLIC_DATA" "$SUPERVISOR_DATA" "$DEV_DATA"; do
  test -f "$path" || { echo "missing CGL1 input: $path" >&2; exit 2; }
done
test ! -e "$OUT" || { echo "refusing existing CGL1 output: $OUT" >&2; exit 2; }
mkdir -p "$OUT"

echo "[diverge-cgl1] node=${SLURMD_NODENAME:-?} backbone=$BACKBONE flip=$FLIP commit=$SOURCE_COMMIT"
nvidia-smi -L
"$PYTHON" -m py_compile \
  "$OVERLAY_ROOT/train/diverge_cgl1_runtime.py" \
  "$OVERLAY_ROOT/train/train_diverge_cgl1.py" \
  "$OVERLAY_ROOT/train/eval_diverge_cgl1.py"
"$PYTHON" "$OVERLAY_ROOT/train/test_diverge_cgl1_runtime.py"

"$PYTHON" "$OVERLAY_ROOT/train/train_diverge_cgl1.py" \
  --base "$PARENT" --base-sha256 "$PARENT_SHA256" \
  --tokenizer "$TOKENIZER" --tokenizer-sha256 "$TOKENIZER_SHA256" \
  --backbone-name "$BACKBONE" \
  --public-data "$PUBLIC_DATA" --public-data-sha256 "$PUBLIC_DATA_SHA256" \
  --supervisor-data "$SUPERVISOR_DATA" \
  --supervisor-data-sha256 "$SUPERVISOR_DATA_SHA256" \
  --output "$MODEL_OUT" --pair-batch-size 32 --learning-rate 0.0001 \
  --consistency-weight 0.25 --seed 2026080702 --device cuda "${EXTRA[@]}"

CHECKPOINT=$MODEL_OUT/checkpoint.pt
CHECKPOINT_SHA256=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
"$PYTHON" "$OVERLAY_ROOT/train/eval_diverge_cgl1.py" \
  --checkpoint "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --base "$PARENT" --base-sha256 "$PARENT_SHA256" \
  --tokenizer "$TOKENIZER" --tokenizer-sha256 "$TOKENIZER_SHA256" \
  --data "$DEV_DATA" --data-sha256 "$DEV_DATA_SHA256" \
  --board-type development --output "$DEV_OUT" --batch-size 32 --device cuda

chmod -R a-w "$OUT"
echo "[diverge-cgl1] completed backbone=$BACKBONE flip=$FLIP"
