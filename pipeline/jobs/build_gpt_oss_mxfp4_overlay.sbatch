#!/bin/bash
#SBATCH --job-name=gpt-oss-mxfp4-overlay
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/gpt_oss_mxfp4_overlay_%j.out
#SBATCH --error=logs/gpt_oss_mxfp4_overlay_%j.err

set -euo pipefail

for variable in PYTHON RUNTIME RUNTIME_MANIFEST_SHA256 OUTPUT_ROOT REPORT; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
[[ -x "$PYTHON" ]]
[[ -d "$RUNTIME" && ! -L "$RUNTIME" ]]
[[ "$OUTPUT_ROOT" == /lustre/fs1/home/sa305415/shohin/artifacts/runtimes/* ]]
[[ "$REPORT" == /lustre/fs1/home/sa305415/shohin/artifacts/runtimes/* ]]
[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]]
[[ ! -e "$REPORT" && ! -L "$REPORT" ]]
if [[ -n "${SOURCE_ROOT:-}" || -n "${SOURCE_REPORT:-}" ]]; then
  [[ -n "${SOURCE_ROOT:-}" && -n "${SOURCE_REPORT:-}" ]]
  [[ -d "$SOURCE_ROOT" && ! -L "$SOURCE_ROOT" ]]
  [[ -f "$SOURCE_REPORT" && ! -L "$SOURCE_REPORT" ]]
  source_overlay_args=(--source-root "$SOURCE_ROOT" --source-report "$SOURCE_REPORT")
else
  source_overlay_args=()
fi

source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
q36_verify_runtime
mkdir -p logs "$(dirname "$OUTPUT_ROOT")"

export PATH=/apps/slurm/current/bin:/usr/bin:/bin
export PYTHONPATH="$RUNTIME/pipeline"
export HF_HOME="$SLURM_TMPDIR/hf"
export XDG_CACHE_HOME="$SLURM_TMPDIR/cache"
export TMPDIR="$SLURM_TMPDIR"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

"$PYTHON" -P -s -B "$RUNTIME/pipeline/build_gpt_oss_mxfp4_overlay.py" \
  --python "$PYTHON" \
  --output-root "$OUTPUT_ROOT" \
  --report "$REPORT" \
  --workers 8 \
  "${source_overlay_args[@]}"

[[ -d "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]]
[[ -f "$OUTPUT_ROOT/SHA256SUMS" && ! -L "$OUTPUT_ROOT/SHA256SUMS" ]]
[[ -f "$REPORT" && ! -L "$REPORT" ]]
(cd "$OUTPUT_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
chmod -R a-w "$OUTPUT_ROOT" "$REPORT"
