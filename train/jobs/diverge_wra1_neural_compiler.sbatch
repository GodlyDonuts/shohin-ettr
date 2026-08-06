#!/bin/bash
#SBATCH -J diverge-wra1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH --exclude=evc33,evc43
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/diverge_wra1_%j.out
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
CPU_REPORT=$CODE_ROOT/artifacts/reasoning/diverge_wra1/cpu_gate_seed202608056500.json
UPDATES=${UPDATES:-1200}
BATCH_SIZE=${BATCH_SIZE:-8}
EVAL_COUNT=${EVAL_COUNT:-256}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-4}
SEED=${SEED:-202608056500}
DATA_SEED=${DATA_SEED:-202608056600}
EVAL_SEED=${EVAL_SEED:-202608056700}

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
test -f "$CPU_REPORT"
test -f "$CODE_ROOT/train/diverge_wra1_neural_compiler.py"
test -f "$CODE_ROOT/train/diverge_wra1_whole_record.py"
test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
test ! -e "${OUT%.json}.pt" || {
  echo "refusing existing checkpoint: ${OUT%.json}.pt" >&2
  exit 2
}
mkdir -p "$(dirname "$OUT")"
trap 'test ! -e "$OUT" || chmod a-w "$OUT"; test ! -e "${OUT%.json}.pt" || chmod a-w "${OUT%.json}.pt"' EXIT

echo "[diverge-wra1] node=${SLURMD_NODENAME:-?} code=$CODE_ROOT commit=$SOURCE_COMMIT"
nvidia-smi -L
"$PYTHON" "$CODE_ROOT/train/test_diverge_wra1_whole_record.py"
"$PYTHON" "$CODE_ROOT/train/test_diverge_wra1_neural_compiler.py"
"$PYTHON" "$CODE_ROOT/train/diverge_wra1_neural_compiler.py" \
  --base "$PARENT" \
  --tokenizer "$TOKENIZER" \
  --sc1-checkpoint "$SC1" \
  --cpu-report "$CPU_REPORT" \
  --output "$OUT" \
  --seed "$SEED" \
  --data-seed "$DATA_SEED" \
  --eval-seed "$EVAL_SEED" \
  --updates "$UPDATES" \
  --batch-size "$BATCH_SIZE" \
  --eval-count "$EVAL_COUNT" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --layer 17 \
  --width 192 \
  --pair-width 64 \
  --slot-layers 2 \
  --slot-heads 4 \
  --learning-rate 0.0003 \
  --threads 4 \
  --device cuda \
  --log-every 100

echo "[diverge-wra1] completed"
