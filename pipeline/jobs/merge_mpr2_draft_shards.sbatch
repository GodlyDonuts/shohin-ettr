#!/bin/bash
#SBATCH --job-name=mpr2-draft-merge
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/mpr2_draft_merge_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/mpr2_draft_merge_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 SHARD_REPORTS OUTPUT REPORT; do test -n "${!variable:-}" || exit 2; done
BASE=/lustre/fs1/home/sa305415/shohin; PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"; (cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
IFS=: read -r -a reports <<<"$SHARD_REPORTS"; args=(); for path in "${reports[@]}"; do test -f "$path"; args+=(--shard-report "$path"); done
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
"$PYTHON" "$RUNTIME/pipeline/merge_mpr2_draft_shards.py" "${args[@]}" --output "$OUTPUT" --report "$REPORT"
chmod a-w "$OUTPUT" "$REPORT"

