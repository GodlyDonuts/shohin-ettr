#!/bin/bash
#SBATCH --job-name=install-bnb
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/install_bnb_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/install_bnb_%j.err

set -euo pipefail
: "${OUTPUT:?OUTPUT is required}"
: "${VERSION:?VERSION is required}"
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
PARTIAL=$OUTPUT.partial.$SLURM_JOB_ID
test -x "$PYTHON"; test ! -e "$OUTPUT"; test ! -e "$PARTIAL"
mkdir -p "$PARTIAL" logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PYTHON" -m pip install --no-deps --target "$PARTIAL" "bitsandbytes==$VERSION"
PYTHONPATH="$PARTIAL" "$PYTHON" - <<PY
import bitsandbytes
if bitsandbytes.__version__ != "$VERSION":
    raise SystemExit("bitsandbytes version differs")
print(bitsandbytes.__version__)
PY
cd "$PARTIAL"
find . -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z | \
  xargs -0 sha256sum > SHA256SUMS
sha256sum -c SHA256SUMS >/dev/null
chmod -R a-w "$PARTIAL"; mv "$PARTIAL" "$OUTPUT"
sha256sum "$OUTPUT/SHA256SUMS"
