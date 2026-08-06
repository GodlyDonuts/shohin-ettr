#!/bin/bash
#SBATCH -J diverge-vmt1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/diverge_vmt1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
SOURCE_COMMIT=${SOURCE_COMMIT:?set immutable SOURCE_COMMIT}
OUTPUT=${OUTPUT:?set isolated output directory}
DATA_SHA256=${DATA_SHA256:?set verified VMT1 board SHA256}
BOARD_REPORT_SHA256=${BOARD_REPORT_SHA256:?set verified VMT1 report SHA256}
MODE=${MODE:-fit}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL=$BASE/artifacts/external/qwen3.5-0.8b-2fc0636
DATA=${DATA:-$BASE/artifacts/reasoning/diverge_vmt1/fit_board_seed2026080602.jsonl}
BOARD_REPORT=${BOARD_REPORT:-$BASE/artifacts/reasoning/diverge_vmt1/fit_board_seed2026080602.report.json}

export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX"
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
test "$(sha256sum "$DATA" | awk '{print $1}')" = "$DATA_SHA256"
test "$(sha256sum "$BOARD_REPORT" | awk '{print $1}')" = "$BOARD_REPORT_SHA256"
test ! -e "$OUTPUT" || { echo "refusing existing output: $OUTPUT" >&2; exit 2; }
mkdir -p "$(dirname "$OUTPUT")" logs
trap 'test ! -e "$OUTPUT" || chmod -R a-w "$OUTPUT"' EXIT

echo "[diverge-vmt1] node=${SLURMD_NODENAME:-?} code=$CODE_ROOT commit=$SOURCE_COMMIT mode=$MODE"
nvidia-smi -L
"$PYTHON" "$CODE_ROOT/train/test_diverge_vmt1_workspace.py"
"$PYTHON" "$CODE_ROOT/train/test_diverge_vmt1_product.py"
"$PYTHON" "$CODE_ROOT/train/test_train_diverge_vmt1.py"
"$PYTHON" -m py_compile \
  "$CODE_ROOT/train/diverge_vmt1_workspace.py" \
  "$CODE_ROOT/train/diverge_vmt1_product.py" \
  "$CODE_ROOT/train/train_diverge_vmt1.py"

COMMON=(
  --model-root "$MODEL"
  --model-source-root "$MODEL"
  --model-revision 2fc06364715b967f1860aea9cf38778875588b17
  --data "$DATA"
  --board-report "$BOARD_REPORT"
  --output "$OUTPUT"
  --expected-rows 16
  --max-sequence-length 1024
  --learning-rate 2e-4
  --lora-layers 4
  --lora-rank 8
  --lora-alpha 16
  --latent-width 384
  --trajectory-slots 8
  --recurrent-steps 8
  --attention-heads 8
  --ff-multiplier 2
  --assignment-temperature 0.1
  --validity-margin 1
  --trace-weight 1
  --validity-weight 0.25
  --halting-weight 0.01
  --seed 2026080602
)

if [[ "$MODE" == smoke ]]; then
  "$PYTHON" "$CODE_ROOT/train/train_diverge_vmt1.py" \
    "${COMMON[@]}" \
    --gate-mode smoke --updates 2 --batch-size 1 --gradient-accumulation 1 \
    --log-interval 1 --checkpoint-interval 2
elif [[ "$MODE" == fit ]]; then
  "$PYTHON" "$CODE_ROOT/train/train_diverge_vmt1.py" \
    "${COMMON[@]}" \
    --gate-mode fit --updates 100 --batch-size 1 --gradient-accumulation 16 \
    --log-interval 10 --checkpoint-interval 100
else
  echo "unknown VMT1 mode: $MODE" >&2
  exit 2
fi

echo "[diverge-vmt1] completed mode=$MODE"
