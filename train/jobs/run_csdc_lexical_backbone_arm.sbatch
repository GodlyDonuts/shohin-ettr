#!/bin/bash
#SBATCH -J csdc-lex-arm
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH --exclude=evc33,evc43
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/csdc_lexical_arm_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
PYTHON=$BASE/miniforge3/bin/python3
BACKBONE=${BACKBONE:?set BACKBONE to shohin or smollm2}
SHUFFLE=${SHUFFLE:-0}
SEED=${SEED:-2026080505}
DATA_ROOT=${DATA_ROOT:-$BASE/artifacts/reasoning/csdc_lexical_backbone_gate_v1}
OUT_ROOT=${OUT_ROOT:-$BASE/train/csdc_lexical_backbone_gate_v1}

case "$BACKBONE" in
  shohin)
    CKPT=$BASE/train/flagship_out/ckpt_0300000.pt
    TOKENIZER=$BASE/artifacts/shohin-tok-32k.json
    DATA=$DATA_ROOT/source_shohin
    ;;
  smollm2)
    CKPT=$BASE/train/ettr_smollm2_control_parent_a2026072801_7881d8e/joint-model-final.pt
    TOKENIZER=${TOKENIZER_SMOL:?set exact SmolLM2 tokenizer path}
    DATA=$DATA_ROOT/smollm2
    ;;
  *) echo "invalid BACKBONE=$BACKBONE" >&2; exit 2 ;;
esac

LABEL=$BACKBONE
EXTRA=()
if [ "$SHUFFLE" = 1 ]; then
  LABEL=${BACKBONE}_shuffled
  EXTRA+=(--shuffle-supervision)
elif [ "$SHUFFLE" != 0 ]; then
  echo "SHUFFLE must be 0 or 1" >&2
  exit 2
fi
OUT=$OUT_ROOT/$LABEL
TRAIN=$DATA/train.jsonl
COMPOSITIONAL=$DATA/development_compositional.jsonl
LEXICAL=$DATA/development_lexical_ood.jsonl
REPORT=$DATA/report.json
ADAPTER=$OUT/compiler.pt

export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE"
for path in "$CKPT" "$TOKENIZER" "$TRAIN" "$COMPOSITIONAL" "$LEXICAL" "$REPORT"; do
  test -f "$path" || { echo "missing input: $path" >&2; exit 2; }
done
test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
mkdir -p "$OUT"

echo "[csdc-lex-arm] backbone=$BACKBONE shuffle=$SHUFFLE node=${SLURMD_NODENAME:-?}"
nvidia-smi -L
"$PYTHON" "$CODE_ROOT/train/train_referential_literal_pointer_compiler.py" \
  --base "$CKPT" --data "$TRAIN" --report "$REPORT" --tokenizer "$TOKENIZER" \
  --out "$ADAPTER" --layer 19 --width 384 --heads 8 --ff 1408 \
  --encoder-layers 5 --role-supervision --role-weight 0.5 --ordinary-tagger \
  --batch-size 64 --epochs 1 --lr 0.001 --warmup 50 --clip 1.0 \
  --kind-weight 1.0 --seed "$SEED" "${EXTRA[@]}"

for SPLIT in development_compositional development_lexical_ood; do
  "$PYTHON" "$CODE_ROOT/train/eval_referential_literal_pointer_compiler.py" \
    --base "$CKPT" --adapter "$ADAPTER" --data "$DATA/$SPLIT.jsonl" \
    --report "$REPORT" --tokenizer "$TOKENIZER" --split "$SPLIT" \
    --out "$OUT/$SPLIT.json" --batch-size 64
done

echo "[csdc-lex-arm] completed $LABEL"
