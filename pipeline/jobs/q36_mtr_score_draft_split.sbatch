#!/bin/bash
#SBATCH --job-name=q36-mtr-score-split
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_mtr_score_split_%j.out
#SBATCH --error=logs/q36_mtr_score_split_%j.err

set -euo pipefail

for variable in PYTHON RUNTIME SCORE_SCRIPT SCORE_SCRIPT_SHA256 CANDIDATES ASSESSOR_BOARD SPLIT OUTPUT; do
  [[ -n "$(printenv "$variable")" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
case "$SPLIT" in train|development) ;; *) exit 2 ;; esac
[[ "$SCORE_SCRIPT_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
[[ "$(sha256sum "$SCORE_SCRIPT" | cut -d' ' -f1)" == "$SCORE_SCRIPT_SHA256" ]] || exit 2
[[ -f "$CANDIDATES" && -f "$ASSESSOR_BOARD" && ! -e "$OUTPUT" ]] || exit 2
tmp=/tmp/q36-mtr-score-split-$SLURM_JOB_ID
[[ ! -e "$tmp" && ! -L "$tmp" ]] || exit 2
mkdir -m 700 "$tmp"
cleanup() {
  [[ "$tmp" =~ ^/tmp/q36-mtr-score-split-[0-9]+$ ]] || return 0
  rm -rf --one-file-system -- "$tmp"
}
trap cleanup EXIT
export SLURM_TMPDIR=$tmp TMPDIR=$tmp
export PATH=/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export PYTHONPATH=$RUNTIME/train:$RUNTIME/pipeline
"$PYTHON" "$SCORE_SCRIPT" \
  --candidates "$CANDIDATES" \
  --assessor-board "$ASSESSOR_BOARD" \
  --output "$OUTPUT" \
  --split "$SPLIT"
test -f "$OUTPUT"
chmod a-w "$OUTPUT"
