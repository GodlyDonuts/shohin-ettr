#!/bin/bash
#SBATCH -J diverge-jet1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/diverge_jet1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
SOURCE_COMMIT=${SOURCE_COMMIT:?set immutable SOURCE_COMMIT}
OUT=${OUT:?set isolated checkpoint path}
REPORT=${REPORT:?set isolated report path}
MODE=${MODE:-scientific}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL=$BASE/artifacts/external/qwen3.5-0.8b-2fc0636

export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE"

test "$(sha256sum "$MODEL/config.json" | awk '{print $1}')" = \
  b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204
test "$(sha256sum "$MODEL/model.safetensors-00001-of-00001.safetensors" | awk '{print $1}')" = \
  04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
test "$(sha256sum "$MODEL/model.safetensors.index.json" | awk '{print $1}')" = \
  d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e
test "$(sha256sum "$MODEL/tokenizer.json" | awk '{print $1}')" = \
  5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42
test "$(sha256sum "$MODEL/tokenizer_config.json" | awk '{print $1}')" = \
  49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c
test ! -e "$OUT" || { echo "refusing existing checkpoint: $OUT" >&2; exit 2; }
test ! -e "$REPORT" || { echo "refusing existing report: $REPORT" >&2; exit 2; }
mkdir -p "$(dirname "$OUT")" "$(dirname "$REPORT")"
trap 'test ! -e "$OUT" || chmod a-w "$OUT"; test ! -e "$REPORT" || chmod a-w "$REPORT"' EXIT

echo "[diverge-jet1] node=${SLURMD_NODENAME:-?} code=$CODE_ROOT commit=$SOURCE_COMMIT mode=$MODE"
nvidia-smi -L
"$PYTHON" "$CODE_ROOT/train/test_diverge_jet1.py"
"$PYTHON" "$CODE_ROOT/train/test_train_diverge_jet1.py"

if [[ "$MODE" == smoke ]]; then
  "$PYTHON" "$CODE_ROOT/train/train_diverge_jet1.py" \
    --model-root "$MODEL" \
    --output "$OUT" \
    --report "$REPORT" \
    --seed 202608058800 \
    --updates 2 \
    --batch-size 2 \
    --eval-batch-size 4 \
    --eval-count 8 \
    --primitive-count 32 \
    --primitive-batch-size 16 \
    --parity-count 8 \
    --warmup 1 \
    --device cuda \
    --smoke
elif [[ "$MODE" == scientific ]]; then
  "$PYTHON" "$CODE_ROOT/train/train_diverge_jet1.py" \
    --model-root "$MODEL" \
    --output "$OUT" \
    --report "$REPORT" \
    --seed 202608058800 \
    --updates 1600 \
    --batch-size 8 \
    --eval-batch-size 8 \
    --eval-count 512 \
    --primitive-count 20000 \
    --primitive-batch-size 512 \
    --parity-count 1000 \
    --device cuda
else
  echo "unknown JET1 mode: $MODE" >&2
  exit 2
fi

echo "[diverge-jet1] completed mode=$MODE"
