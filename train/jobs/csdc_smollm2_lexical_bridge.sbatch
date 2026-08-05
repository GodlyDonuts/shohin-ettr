#!/bin/bash
#SBATCH -J csdc-smol-bridge
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH --exclude=evc33,evc43
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/csdc_smollm2_bridge_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
PYTHON=$BASE/miniforge3/bin/python3
PARENT=$BASE/train/ettr_smollm2_control_parent_a2026072801_7881d8e/joint-model-final.pt
TOKENIZER=$BASE/artifacts/external/smollm2_135m_instruct_83212e1e/tokenizer.json
WARM=$BASE/train/csdc_lexical_backbone_gate_v1/smollm2/compiler.pt
REASONER=$BASE/artifacts/reasoning/learned_pspa_pilot_8fb5d61/seed47.pt
OUT_ROOT=${OUT_ROOT:-$BASE/train/csdc_smollm2_lexical_bridge_v1}
OUT=$OUT_ROOT/seed2026080507.json

export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE"
for path in "$PARENT" "$TOKENIZER" "$WARM" "$REASONER"; do
  test -f "$path" || { echo "missing input: $path" >&2; exit 2; }
done
test "$(sha256sum "$WARM" | awk '{print $1}')" = \
  abd22528da0d8dc4718c7a89d9c94520540a2f38b7f0b1d9a9e623d0af23cf4d
test "$(sha256sum "$REASONER" | awk '{print $1}')" = \
  c374e3b566808cb317ffcd2725653c9073d2e7aebeb75e93ed7ea2a7e2e27044
test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
test ! -e "${OUT%.json}.pt" || {
  echo "refusing existing checkpoint: ${OUT%.json}.pt" >&2
  exit 2
}
mkdir -p "$OUT_ROOT"

echo "[csdc-smol-bridge] node=${SLURMD_NODENAME:-?} code=$CODE_ROOT"
nvidia-smi -L
"$PYTHON" "$CODE_ROOT/train/csdc_smollm2_lexical_bridge.py" \
  --base "$PARENT" \
  --tokenizer "$TOKENIZER" \
  --warm-adapter "$WARM" \
  --reasoner-checkpoint "$REASONER" \
  --output "$OUT" \
  --seed 2026080507 \
  --updates 1500 \
  --batch-size 128 \
  --eval-count 1024 \
  --eval-batch-size 64 \
  --learning-rate 0.0003 \
  --weight-decay 0.01 \
  --train-templates 0,1,2

chmod -R a-w "$OUT_ROOT"
echo "[csdc-smol-bridge] completed"

