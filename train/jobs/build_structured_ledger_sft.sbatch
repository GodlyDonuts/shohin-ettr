#!/bin/bash
#SBATCH -J ledger-sft
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/structured_ledger_sft_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
OUT=${OUT:?set immutable output root}
SOURCE=$BASE/artifacts/product_reasoning/structured_ledger_rg_v4_r1
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
"$PYTHON" "$CODE_ROOT/pipeline/materialize_structured_ledger_sft.py" \
  --train-source "$SOURCE/train.jsonl" \
  --development-source "$SOURCE/development.jsonl" \
  --expected-train-sha256 02bd92260538ed75c7b1e82aa7cdb0bed4029ed06c8e982ef1b059e5605699df \
  --expected-development-sha256 f5d4a7150c384da046b3e86c76f9b73e1fdc15d4b56add29c3cf0fa83450bdee \
  --output "$OUT"
find "$OUT" -type f -exec chmod a-w {} +
