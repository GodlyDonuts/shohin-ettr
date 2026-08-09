#!/bin/bash
#SBATCH --job-name=ecr1-data
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/ecr1_data_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/ecr1_data_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 DATA DATA_SHA256 CUSTODY \
  CUSTODY_SHA256 OUTPUT REPORT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$DATA" | cut -d' ' -f1)" = "$DATA_SHA256"
test "$(sha256sum "$CUSTODY" | cut -d' ' -f1)" = "$CUSTODY_SHA256"
test ! -e "$OUTPUT" && test ! -e "$REPORT"
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
"$PYTHON" "$RUNTIME/pipeline/build_ecr1_complete_training.py" \
  --data "$DATA" --data-sha256 "$DATA_SHA256" \
  --custody "$CUSTODY" --custody-sha256 "$CUSTODY_SHA256" \
  --output "$OUTPUT" --report "$REPORT"
sha256sum "$OUTPUT" "$REPORT"
chmod a-w "$OUTPUT" "$REPORT"
