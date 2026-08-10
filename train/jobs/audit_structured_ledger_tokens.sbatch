#!/bin/bash
#SBATCH -J ledger-tok
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/structured_ledger_tokens_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
DATA=${DATA:?set immutable SFT data root}
OUT=${OUT:?set immutable output report}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL=$BASE/artifacts/external/qwen3.5-0.8b-2fc0636

test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
"$PYTHON" "$CODE_ROOT/pipeline/audit_ledger_token_geometry.py" \
  --model-root "$MODEL" \
  --model-revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --train "$DATA/train.jsonl" \
  --development "$DATA/development.jsonl" \
  --expected-train-sha256 "${TRAIN_SHA256:?set train SHA-256}" \
  --expected-development-sha256 "${DEVELOPMENT_SHA256:?set development SHA-256}" \
  --max-sequence-length "${MAX_SEQUENCE_LENGTH:-1024}" \
  --output "$OUT"
chmod a-w "$OUT"
