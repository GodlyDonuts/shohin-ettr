#!/bin/bash
#SBATCH -J ltr1-audit
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/ltr1_audit_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
OUT=${OUT:?set isolated output report}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
DATA=$BASE/artifacts/product_reasoning/cte1/data_70d6a76_r4/development.jsonl
REPORT=$BASE/artifacts/product_reasoning/cte1/eval_1e21f38_r1/normal.json

test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
export PYTHONPATH="$CODE_ROOT/pipeline"
"$PYTHON" "$CODE_ROOT/pipeline/audit_ltr1_record_locality.py" \
  --data "$DATA" \
  --expected-data-sha256 aff466172c74dd7d13a183d117e32ec10d5da2048d10253189e4e3b3a599eb04 \
  --report "$REPORT" \
  --expected-report-sha256 8c75ca704cb26df4ca4ad084f6a4b9e9f5f5a1c1ac3c426326a11b4af0e12954 \
  --output "$OUT"
chmod a-w "$OUT"
