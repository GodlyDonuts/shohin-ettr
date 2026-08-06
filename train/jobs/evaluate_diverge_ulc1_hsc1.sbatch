#!/bin/bash
#SBATCH -J ulc1-hsc1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/ulc1_hsc1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
SOURCE_COMMIT=${SOURCE_COMMIT:?set immutable SOURCE_COMMIT}
OUT=${OUT:?set isolated output JSON path}
PYTHON=$BASE/miniforge3/bin/python3
PARENT=$BASE/train/ettr_smollm2_control_parent_a2026072801_7881d8e/joint-model-final.pt
TOKENIZER=$BASE/artifacts/external/smollm2_135m_instruct_83212e1e/tokenizer.json
SC1=$BASE/artifacts/reasoning/diverge_sc1/neural_seed202608056200.pt
HSC1=$BASE/artifacts/reasoning/diverge_hsc1/neural_seed202608056900.pt
COUNT=${COUNT:-64}
BATCH_SIZE=${BATCH_SIZE:-4}

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
test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
mkdir -p "$(dirname "$OUT")"
trap 'test ! -e "$OUT" || chmod a-w "$OUT"' EXIT

echo "[ulc1-hsc1] node=${SLURMD_NODENAME:-?} code=$CODE_ROOT commit=$SOURCE_COMMIT"
nvidia-smi -L
"$PYTHON" "$CODE_ROOT/train/test_diverge_ulc1_mdd.py"
"$PYTHON" "$CODE_ROOT/train/test_evaluate_diverge_ulc1_hsc1.py"
"$PYTHON" "$CODE_ROOT/train/evaluate_diverge_ulc1_hsc1.py" \
  --base "$PARENT" \
  --tokenizer "$TOKENIZER" \
  --sc1-checkpoint "$SC1" \
  --hsc1-checkpoint "$HSC1" \
  --output "$OUT" \
  --seed 202608057600 \
  --count "$COUNT" \
  --batch-size "$BATCH_SIZE" \
  --component-k 2 \
  --max-nodes 1000000 \
  --max-groups 100000 \
  --device cuda

echo "[ulc1-hsc1] completed"
