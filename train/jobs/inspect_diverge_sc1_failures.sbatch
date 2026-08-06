#!/bin/bash
#SBATCH -J diverge-sc1-audit
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH --exclude=evc33,evc43
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/diverge_sc1_audit_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
SOURCE_COMMIT=${SOURCE_COMMIT:?set immutable SOURCE_COMMIT}
OUT=${OUT:?set isolated output JSON path}
PYTHON=$BASE/miniforge3/bin/python3
PARENT=$BASE/train/ettr_smollm2_control_parent_a2026072801_7881d8e/joint-model-final.pt
TOKENIZER=$BASE/artifacts/external/smollm2_135m_instruct_83212e1e/tokenizer.json
CHECKPOINT=$BASE/artifacts/reasoning/diverge_sc1/neural_seed202608056200.pt

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
test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = \
  7b5348cacb1772bf45e34442e94010db71a6be20bd8d689477d037ac5fee2ffd
test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
mkdir -p "$(dirname "$OUT")"
trap 'test ! -e "$OUT" || chmod a-w "$OUT"' EXIT

echo "[diverge-sc1-audit] node=${SLURMD_NODENAME:-?} code=$CODE_ROOT commit=$SOURCE_COMMIT"
nvidia-smi -L
"$PYTHON" "$CODE_ROOT/train/inspect_diverge_sc1_failures.py" \
  --base "$PARENT" \
  --tokenizer "$TOKENIZER" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUT" \
  --count 32 \
  --seed 202608056400 \
  --batch-size 4 \
  --layer 17 \
  --width 192 \
  --pair-width 64 \
  --threads 4 \
  --device cuda

echo "[diverge-sc1-audit] completed"
