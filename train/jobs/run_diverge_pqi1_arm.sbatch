#!/bin/bash
#SBATCH -J diverge-pqi1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH --exclude=evc33,evc43
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/diverge_pqi1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable PQI1 code root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set immutable PQI1 source commit}
BACKBONE=${BACKBONE:?set shohin or smollm2}
SHUFFLE=${SHUFFLE:-0}
OUT=${OUT:?set isolated PQI1 output root}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
QUERY_DATA=$BASE/artifacts/reasoning/diverge_rrg1/data_8246fcb_r1/query_train.jsonl
QUERY_DATA_SHA256=2d325c860e707307886f782350e7ec35ae8c23ae275260b0a937bbb738078c1c
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
if [ "$SHUFFLE" = 1 ]; then
  EXTRA+=(--shuffle-supervision)
elif [ "$SHUFFLE" != 0 ]; then
  echo "SHUFFLE must be 0 or 1" >&2
  exit 2
fi

MODEL_OUT=$OUT/model
DEV_OUT=$OUT/development.json
export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs"
cd "$BASE"

for path in "$PARENT" "$TOKENIZER" "$QUERY_DATA" "$DEV_DATA"; do
  test -f "$path" || { echo "missing PQI1 input: $path" >&2; exit 2; }
done
test ! -e "$OUT" || { echo "refusing existing PQI1 output: $OUT" >&2; exit 2; }
mkdir -p "$OUT"

echo "[diverge-pqi1] node=${SLURMD_NODENAME:-?} backbone=$BACKBONE shuffle=$SHUFFLE commit=$SOURCE_COMMIT"
nvidia-smi -L
"$PYTHON" -m py_compile \
  "$CODE_ROOT/train/diverge_pqi1_runtime.py" \
  "$CODE_ROOT/train/train_diverge_pqi1.py" \
  "$CODE_ROOT/train/eval_diverge_pqi1.py"
"$PYTHON" "$CODE_ROOT/train/test_diverge_pqi1_runtime.py"

"$PYTHON" "$CODE_ROOT/train/train_diverge_pqi1.py" \
  --base "$PARENT" --base-sha256 "$PARENT_SHA256" \
  --tokenizer "$TOKENIZER" --tokenizer-sha256 "$TOKENIZER_SHA256" \
  --backbone-name "$BACKBONE" \
  --query-data "$QUERY_DATA" --query-data-sha256 "$QUERY_DATA_SHA256" \
  --output "$MODEL_OUT" --batch-size 128 --learning-rate 0.001 \
  --seed 2026080631 --device cuda "${EXTRA[@]}"

CHECKPOINT=$MODEL_OUT/checkpoint.pt
CHECKPOINT_SHA256=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
"$PYTHON" "$CODE_ROOT/train/eval_diverge_pqi1.py" \
  --checkpoint "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --base "$PARENT" --base-sha256 "$PARENT_SHA256" \
  --tokenizer "$TOKENIZER" --tokenizer-sha256 "$TOKENIZER_SHA256" \
  --data "$DEV_DATA" --data-sha256 "$DEV_DATA_SHA256" \
  --board-type development --output "$DEV_OUT" --batch-size 128 --device cuda

chmod -R a-w "$OUT"
echo "[diverge-pqi1] completed backbone=$BACKBONE shuffle=$SHUFFLE"
