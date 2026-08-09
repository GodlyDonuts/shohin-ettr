#!/bin/bash
#SBATCH --job-name=ecr1-custody
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/ecr1_custody_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/ecr1_custody_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT DATA DATA_SHA256 OUTPUT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$DATA" | cut -d' ' -f1)" = "$DATA_SHA256"
test ! -e "$OUTPUT"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
"$PYTHON" "$RUNTIME/pipeline/audit_ecr1_sequence_custody.py" \
  --model-root "$MODEL_ROOT" --data "$DATA" --data-sha256 "$DATA_SHA256" \
  --output "$OUTPUT" --rows "${ROWS:-9655}" \
  --data-seed "${DATA_SEED:-2026080814}" \
  --max-sequence-length "${MAX_LENGTH:-4096}"
sha256sum "$OUTPUT"
chmod a-w "$OUTPUT"
