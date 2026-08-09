#!/bin/bash
#SBATCH --job-name=hf-pinned-download
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/hf_pinned_download_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/hf_pinned_download_%j.err

set -euo pipefail

: "${REPO_ID:?REPO_ID is required}"
: "${REVISION:?REVISION is required}"
: "${OUTPUT:?OUTPUT is required}"

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
PARTIAL="$OUTPUT.partial.$SLURM_JOB_ID"

test -x "$PYTHON"
test ! -e "$OUTPUT"
test ! -e "$PARTIAL"
mkdir -p "$PARTIAL" logs

set -a
source "$BASE/.env" 2>/dev/null || true
set +a
export REPO_ID REVISION PARTIAL

"$PYTHON" - <<'PY'
import os

from huggingface_hub import snapshot_download


snapshot_download(
    repo_id=os.environ["REPO_ID"],
    revision=os.environ["REVISION"],
    local_dir=os.environ["PARTIAL"],
    token=(
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    ),
)
PY

cd "$PARTIAL"
printf '%s\n' "$REVISION" > SOURCE_REVISION
find . -type f ! -name SHA256SUMS -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 sha256sum > SHA256SUMS
sha256sum -c SHA256SUMS >/dev/null
chmod -R a-w "$PARTIAL"
mv "$PARTIAL" "$OUTPUT"
sha256sum "$OUTPUT/config.json" "$OUTPUT/SHA256SUMS"
