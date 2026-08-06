#!/bin/bash
#SBATCH -J diverge-ltm1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/diverge_ltm1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
SOURCE_COMMIT=${SOURCE_COMMIT:?set immutable SOURCE_COMMIT}
OUTPUT=${OUTPUT:?set isolated output directory}
MODE=${MODE:-fit}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL=$BASE/artifacts/external/qwen3.5-0.8b-2fc0636
DATA=${DATA:-$BASE/artifacts/product_reasoning/data/v10_tokenbalanced_35m20c10s10p25t_4m_verified_r1.jsonl}

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
test "$(sha256sum "$DATA" | awk '{print $1}')" = \
  2461d6f70b44a142854d56c24e1fb42d600065e5788a2c4e055ba47b12696549
test ! -e "$OUTPUT" || { echo "refusing existing output: $OUTPUT" >&2; exit 2; }
mkdir -p "$(dirname "$OUTPUT")" logs
trap 'test ! -e "$OUTPUT" || chmod -R a-w "$OUTPUT"' EXIT

echo "[diverge-ltm1] node=${SLURMD_NODENAME:-?} code=$CODE_ROOT commit=$SOURCE_COMMIT mode=$MODE"
nvidia-smi -L
"$PYTHON" "$CODE_ROOT/train/test_diverge_ltm1_workspace.py"
"$PYTHON" -m py_compile \
  "$CODE_ROOT/train/diverge_ltm1_workspace.py" \
  "$CODE_ROOT/train/diverge_ltm1_product.py" \
  "$CODE_ROOT/train/train_diverge_ltm1.py"

COMMON=(
  --model-root "$MODEL"
  --model-source-root "$MODEL"
  --model-revision 2fc06364715b967f1860aea9cf38778875588b17
  --data "$DATA"
  --output "$OUTPUT"
  --learning-rate 2e-4
  --lora-layers 4
  --lora-rank 8
  --lora-alpha 16
  --latent-width 384
  --trajectory-slots 8
  --recurrent-steps 8
  --fault-bits 2
  --attention-heads 8
  --ff-multiplier 2
  --trace-weight 0.25
  --balance-weight 0.01
  --halting-weight 0.01
  --seed 2026080601
  --data-seed 20260802
)

if [[ "$MODE" == smoke ]]; then
  "$PYTHON" "$CODE_ROOT/train/train_diverge_ltm1.py" \
    "${COMMON[@]}" \
    --updates 2 --batch-size 1 --gradient-accumulation 1 \
    --max-rows 4 --max-sequence-length 512 --log-interval 1 \
    --checkpoint-interval 2
elif [[ "$MODE" == fit ]]; then
  "$PYTHON" "$CODE_ROOT/train/train_diverge_ltm1.py" \
    "${COMMON[@]}" \
    --updates 100 --batch-size 1 --gradient-accumulation 16 \
    --max-rows 16 --max-sequence-length 1024 --log-interval 10 \
    --checkpoint-interval 100
elif [[ "$MODE" == broad ]]; then
  "$PYTHON" "$CODE_ROOT/train/train_diverge_ltm1.py" \
    "${COMMON[@]}" \
    --updates 200 --batch-size 1 --gradient-accumulation 16 \
    --max-rows 100000 --max-sequence-length 1024 --log-interval 10 \
    --checkpoint-interval 100
else
  echo "unknown LTM1 mode: $MODE" >&2
  exit 2
fi

echo "[diverge-ltm1] completed mode=$MODE"
