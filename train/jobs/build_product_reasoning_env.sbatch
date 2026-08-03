#!/bin/bash
#SBATCH --job-name=build-reasoning-env
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/build_reasoning_env_%j.out
#SBATCH --error=logs/build_reasoning_env_%j.err

set -euo pipefail

: "${ENV_ROOT:?ENV_ROOT is required}"
: "${TRANSFORMERS_COMMIT:?TRANSFORMERS_COMMIT is required}"

BASE_PYTHON=${BASE_PYTHON:-/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python3.13}
test -x "$BASE_PYTHON"
test ! -e "$ENV_ROOT"

mkdir -p logs "$(dirname "$ENV_ROOT")"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

trap 'touch "$ENV_ROOT/FAILED" 2>/dev/null || true' ERR
"$BASE_PYTHON" -m venv --system-site-packages "$ENV_ROOT"
"$ENV_ROOT/bin/python" -m pip install \
  "git+https://github.com/huggingface/transformers.git@${TRANSFORMERS_COMMIT}" \
  "torchvision==0.21.0" \
  accelerate \
  peft \
  sentencepiece

"$ENV_ROOT/bin/python" - <<'PY' > "$ENV_ROOT/runtime.json"
import json
import platform

import torch
import transformers

payload = {
    "schema": "shohin-product-reasoning-environment-v1",
    "hostname": platform.node(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "cuda_available_during_build": torch.cuda.is_available(),
}
print(json.dumps(payload, sort_keys=True))
PY
"$ENV_ROOT/bin/python" -m pip freeze --all > "$ENV_ROOT/pip-freeze.txt"
sha256sum "$ENV_ROOT/runtime.json" "$ENV_ROOT/pip-freeze.txt" \
  > "$ENV_ROOT/SHA256SUMS"
touch "$ENV_ROOT/READY"
rm -f "$ENV_ROOT/FAILED"
