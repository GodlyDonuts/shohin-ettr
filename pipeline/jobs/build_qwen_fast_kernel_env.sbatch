#!/bin/bash
#SBATCH --job-name=qwen-fast-kernel-env
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/qwen_fast_kernel_env_%j.out
#SBATCH --error=logs/qwen_fast_kernel_env_%j.err

set -euo pipefail

: "${BASE_ENV:?BASE_ENV is required}"
: "${OUTPUT_ENV:?OUTPUT_ENV is required}"
FLA_VERSION=${FLA_VERSION:-0.3.2}
FLA_CORE_VERSION=${FLA_CORE_VERSION:-$FLA_VERSION}
EINOPS_VERSION=${EINOPS_VERSION:-0.8.2}
CAUSAL_CONV_VERSION=${CAUSAL_CONV_VERSION:-1.5.0.post8}

test -x "$BASE_ENV/bin/python"
test ! -e "$OUTPUT_ENV"
TEMP_ENV="$OUTPUT_ENV.tmp.$SLURM_JOB_ID"
test ! -e "$TEMP_ENV"

module load cuda/cuda-12.4.0
command -v nvcc
mkdir -p logs "$(dirname "$OUTPUT_ENV")"
cp -al "$BASE_ENV" "$TEMP_ENV"
export MAX_JOBS=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PIP_NO_CACHE_DIR=1

"$TEMP_ENV/bin/python" -m pip install \
  --no-deps \
  "flash-linear-attention==$FLA_VERSION" \
  "fla-core==$FLA_CORE_VERSION" \
  "einops==$EINOPS_VERSION"
"$TEMP_ENV/bin/python" -m pip install \
  --no-deps --no-build-isolation "causal-conv1d==$CAUSAL_CONV_VERSION"
"$TEMP_ENV/bin/python" - <<'PY'
import causal_conv1d
import fla
import torch
from fla.modules import FusedRMSNormGated
from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)
from importlib.metadata import version

assert FusedRMSNormGated is not None
assert chunk_gated_delta_rule is not None
assert fused_recurrent_gated_delta_rule is not None

print(
    {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "fla": version("flash-linear-attention"),
        "fla_core": version("fla-core"),
        "einops": version("einops"),
        "causal_conv1d": getattr(causal_conv1d, "__version__", "unknown"),
    },
    flush=True,
)
PY

mv "$TEMP_ENV" "$OUTPUT_ENV"
chmod -R a-w "$OUTPUT_ENV"
