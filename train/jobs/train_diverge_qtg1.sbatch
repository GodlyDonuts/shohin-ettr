#!/bin/bash
#SBATCH -J diverge-qtg1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/diverge_qtg1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
SOURCE_COMMIT=${SOURCE_COMMIT:?set immutable SOURCE_COMMIT}
OUT=${OUT:?set isolated checkpoint path}
REPORT=${REPORT:?set isolated report path}
PYTHON=$BASE/miniforge3/bin/python3
PARENT=$BASE/train/ettr_smollm2_control_parent_a2026072801_7881d8e/joint-model-final.pt
TOKENIZER=$BASE/artifacts/external/smollm2_135m_instruct_83212e1e/tokenizer.json
SC1=$BASE/artifacts/reasoning/diverge_sc1/neural_seed202608056200.pt
HSC1=$BASE/artifacts/reasoning/diverge_hsc1/neural_seed202608056900.pt
MEI1=$BASE/artifacts/reasoning/diverge_mei1/neural_seed202608057800.pt

export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE"

test "$(sha256sum "$PARENT" | awk '{print $1}')" = \
  8196f810a31e0abe7f3bf0eae0a37b103195f109b7a8e962c7b74b5710c98a02
test "$(sha256sum "$TOKENIZER" | awk '{print $1}')" = \
  9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c
test "$(sha256sum "$SC1" | awk '{print $1}')" = \
  7b5348cacb1772bf45e34442e94010db71a6be20bd8d689477d037ac5fee2ffd
test "$(sha256sum "$HSC1" | awk '{print $1}')" = \
  34c7eaee885ba5201e6e07335add1737b7b7d26b2709861b7967e0b97be64a05
test "$(sha256sum "$MEI1" | awk '{print $1}')" = \
  bed9abefa2ecd2401c11515fe182d89871ab537bd8f8716bd5688ae693b79c29
test ! -e "$OUT" || { echo "refusing existing checkpoint: $OUT" >&2; exit 2; }
test ! -e "$REPORT" || { echo "refusing existing report: $REPORT" >&2; exit 2; }
mkdir -p "$(dirname "$OUT")" "$(dirname "$REPORT")"
trap 'test ! -e "$OUT" || chmod a-w "$OUT"; test ! -e "$REPORT" || chmod a-w "$REPORT"' EXIT

echo "[diverge-qtg1] node=${SLURMD_NODENAME:-?} code=$CODE_ROOT commit=$SOURCE_COMMIT"
nvidia-smi -L
"$PYTHON" "$CODE_ROOT/train/test_diverge_qtg1.py"
"$PYTHON" "$CODE_ROOT/train/train_diverge_qtg1.py" \
  --base "$PARENT" \
  --tokenizer "$TOKENIZER" \
  --sc1-checkpoint "$SC1" \
  --hsc1-checkpoint "$HSC1" \
  --mei1-checkpoint "$MEI1" \
  --output "$OUT" \
  --report "$REPORT" \
  --seed 202608058400 \
  --updates 1600 \
  --batch-size 64 \
  --eval-batch-size 64 \
  --eval-count 20000 \
  --reset-eval-count 2000 \
  --parity-count 10000 \
  --device cuda

echo "[diverge-qtg1] completed"
