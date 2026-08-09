#!/bin/bash
#SBATCH --job-name=mpr2-data
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/mpr2_data_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/mpr2_data_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT TRAIN TRAIN_REPORT TRAIN_DRAFTS TRAIN_DRAFT_REPORT DEVELOPMENT DEVELOPMENT_REPORT DEVELOPMENT_OWNER_CANDIDATES DEVELOPMENT_OWNER_REPORT OUTPUT; do test -n "${!variable:-}" || exit 2; done
BASE=/lustre/fs1/home/sa305415/shohin; PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test ! -e "$OUTPUT"; test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"; (cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
export TOKENIZERS_PARALLELISM=false PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
"$PYTHON" "$RUNTIME/pipeline/build_mpr2_revision_data.py" --train "$TRAIN" --train-report "$TRAIN_REPORT" --train-drafts "$TRAIN_DRAFTS" --train-draft-report "$TRAIN_DRAFT_REPORT" --development "$DEVELOPMENT" --development-report "$DEVELOPMENT_REPORT" --development-owner-candidates "$DEVELOPMENT_OWNER_CANDIDATES" --development-owner-report "$DEVELOPMENT_OWNER_REPORT" --model-root "$MODEL_ROOT" --output "$OUTPUT"
chmod -R a-w "$OUTPUT"

