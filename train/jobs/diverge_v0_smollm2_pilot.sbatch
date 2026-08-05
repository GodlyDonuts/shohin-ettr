#!/bin/bash
#SBATCH -J diverge-smol-pilot
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH --exclude=evc33,evc43
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/diverge_smollm2_pilot_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
SOURCE_COMMIT=${SOURCE_COMMIT:?set immutable SOURCE_COMMIT}
OUT=${OUT:?set isolated output JSON path}
PYTHON=$BASE/miniforge3/bin/python3
PARENT=$BASE/train/ettr_smollm2_control_parent_a2026072801_7881d8e/joint-model-final.pt
TOKENIZER=$BASE/artifacts/external/smollm2_135m_instruct_83212e1e/tokenizer.json
UPDATES=${UPDATES:-600}
BATCH_SIZE=${BATCH_SIZE:-16}
EVAL_COUNT=${EVAL_COUNT:-256}
SEED=${SEED:-2026080517}
DATA_SEED=${DATA_SEED:-202608052000}
EVAL_SEED=${EVAL_SEED:-202608053000}

export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE"

test -f "$PARENT" || { echo "missing parent: $PARENT" >&2; exit 2; }
test -f "$TOKENIZER" || { echo "missing tokenizer: $TOKENIZER" >&2; exit 2; }
test "$(sha256sum "$PARENT" | awk '{print $1}')" = \
  8196f810a31e0abe7f3bf0eae0a37b103195f109b7a8e962c7b74b5710c98a02
test "$(sha256sum "$TOKENIZER" | awk '{print $1}')" = \
  9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c
test -f "$CODE_ROOT/train/diverge_v0_neural_pilot.py"
test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
test ! -e "${OUT%.json}.pt" || {
  echo "refusing existing checkpoint: ${OUT%.json}.pt" >&2
  exit 2
}
mkdir -p "$(dirname "$OUT")"

echo "[diverge-smol-pilot] node=${SLURMD_NODENAME:-?} code=$CODE_ROOT commit=$SOURCE_COMMIT"
nvidia-smi -L
"$PYTHON" "$CODE_ROOT/train/diverge_v0_neural_pilot.py" \
  --base "$PARENT" \
  --tokenizer "$TOKENIZER" \
  --layer 17 \
  --output "$OUT" \
  --seed "$SEED" \
  --data-seed "$DATA_SEED" \
  --eval-seed "$EVAL_SEED" \
  --updates "$UPDATES" \
  --batch-size "$BATCH_SIZE" \
  --eval-count "$EVAL_COUNT" \
  --width 192 \
  --char-width 48 \
  --selection-policy adaptive-gap \
  --learning-rate 0.0003 \
  --threads 4 \
  --device cuda \
  --log-every 100

chmod a-w "$OUT" "${OUT%.json}.pt"
echo "[diverge-smol-pilot] completed"
