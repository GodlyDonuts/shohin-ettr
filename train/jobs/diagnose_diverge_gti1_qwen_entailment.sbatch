#!/bin/bash
#SBATCH -J gti1-qwen-rte
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH --exclude=evc33,evc43
#SBATCH -o /lustre/fs1/home/%u/shohin/logs/gti1_qwen_rte_%j.out
#SBATCH --open-mode=append
set -euo pipefail

ROOT=/lustre/fs1/home/sa305415/shohin
BASE_CODE_ROOT=${BASE_CODE_ROOT:?set immutable qualified base code root}
OVERLAY_ROOT=${OVERLAY_ROOT:?set immutable Qwen attribution overlay}
OUTPUT=${OUTPUT:?set isolated Qwen attribution result}
PYTHON=${PYTHON:-$ROOT/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL=$ROOT/artifacts/external/qwen3.5-0.8b-2fc0636
DATA=$ROOT/artifacts/reasoning/diverge_ccr1/data_ec6d2f3_r1/confirmation_board.jsonl
DATA_SHA256=299237068f436ba33a68487b5300fcd724f8c98bd8bfe6b1916a4ebc7541ebf7

export PYTHONPATH="$OVERLAY_ROOT/train:$BASE_CODE_ROOT/train:$BASE_CODE_ROOT/pipeline"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$ROOT/logs"
cd "$BASE_CODE_ROOT"

test "$(sha256sum "$MODEL/config.json" | awk '{print $1}')" = \
  b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204
test "$(sha256sum "$MODEL/model.safetensors-00001-of-00001.safetensors" | awk '{print $1}')" = \
  04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
test "$(sha256sum "$MODEL/tokenizer.json" | awk '{print $1}')" = \
  5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42
test ! -e "$OUTPUT" || { echo "refusing existing Qwen attribution" >&2; exit 2; }

echo "[gti1-qwen-rte] node=${SLURMD_NODENAME:-?}"
nvidia-smi -L
"$PYTHON" "$OVERLAY_ROOT/train/diagnose_diverge_gti1_qwen_entailment.py" \
  --model-root "$MODEL" \
  --data "$DATA" \
  --data-sha256 "$DATA_SHA256" \
  --output "$OUTPUT" \
  --batch-size 32
