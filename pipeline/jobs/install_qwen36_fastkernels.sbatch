#!/bin/bash
#SBATCH --job-name=q36-kernels
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/q36_kernels_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/q36_kernels_%j.err

set -euo pipefail
: "${OUTPUT:?OUTPUT is required}"
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
PARTIAL=$OUTPUT.partial.$SLURM_JOB_ID
test -x "$PYTHON"; test ! -e "$OUTPUT"; test ! -e "$PARTIAL"
mkdir -p "$PARTIAL" logs
module load cuda/cuda-12.4.0
command -v nvcc
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MAX_JOBS=$SLURM_CPUS_PER_TASK TORCH_CUDA_ARCH_LIST=9.0
export CAUSAL_CONV1D_FORCE_BUILD=TRUE

"$PYTHON" -m pip install --no-deps --target "$PARTIAL" \
  einops==0.8.1 ninja==1.13.0 fla-core==0.4.2 flash-linear-attention==0.4.2
PYTHONPATH="$PARTIAL" "$PYTHON" -m pip install --no-deps \
  --no-build-isolation --no-cache-dir --no-binary causal-conv1d \
  --target "$PARTIAL" causal-conv1d==1.6.2.post1

PYTHONPATH="$PARTIAL" "$PYTHON" - <<'PY'
import causal_conv1d
import fla
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable for Qwen3.6 kernel validation")
print("causal_conv1d", causal_conv1d.__version__)
print("fla", fla.__version__)
print("cuda", torch.cuda.get_device_name(0))
PY
cd "$PARTIAL"
find . -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z | \
  xargs -0 sha256sum > SHA256SUMS
sha256sum -c SHA256SUMS >/dev/null
chmod -R a-w "$PARTIAL"; mv "$PARTIAL" "$OUTPUT"
sha256sum "$OUTPUT/SHA256SUMS"
