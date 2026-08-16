#!/bin/bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36-mtr-synthesis-data-%j.out
#SBATCH --error=logs/q36-mtr-synthesis-data-%j.err

set -euo pipefail

for name in PYTHON RUNTIME BUILDER_SCRIPT BUILDER_SCRIPT_SHA256 TRAINING_ROWS OUTPUT REPORT; do
  [[ -n "${!name:-}" ]] || { printf 'q36 synthesis data missing %s\n' "$name" >&2; exit 2; }
done
[[ -x "$PYTHON" && -f "$BUILDER_SCRIPT" && ! -L "$BUILDER_SCRIPT" ]] || { printf 'q36 synthesis builder differs\n' >&2; exit 2; }
[[ "$(sha256sum "$BUILDER_SCRIPT" | awk '{print $1}')" == "$BUILDER_SCRIPT_SHA256" ]] || { printf 'q36 synthesis builder hash differs\n' >&2; exit 2; }
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" && ! -e "$REPORT" && ! -L "$REPORT" ]] || { printf 'q36 synthesis data output exists\n' >&2; exit 2; }
export PYTHONPATH="$(dirname "$BUILDER_SCRIPT"):$RUNTIME/train"
"$PYTHON" -P -s -B "$BUILDER_SCRIPT" --training-rows "$TRAINING_ROWS" --output "$OUTPUT" --report "$REPORT"
chmod a-w "$OUTPUT" "$REPORT"
