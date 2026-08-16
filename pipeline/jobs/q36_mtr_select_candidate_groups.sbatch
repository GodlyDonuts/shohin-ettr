#!/bin/bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36-mtr-group-select-%j.out
#SBATCH --error=logs/q36-mtr-group-select-%j.err

set -euo pipefail

for name in PYTHON SELECT_SCRIPT SELECT_SCRIPT_SHA256 FIRST_CANDIDATES SECOND_CANDIDATES FIRST_LABEL SECOND_LABEL SPLIT OUTPUT REPORT; do
  [[ -n "${!name:-}" ]] || { printf 'q36 group selection missing %s\n' "$name" >&2; exit 2; }
done
[[ -x "$PYTHON" && -f "$SELECT_SCRIPT" && ! -L "$SELECT_SCRIPT" ]] || { printf 'q36 group selection executable differs\n' >&2; exit 2; }
[[ "$(sha256sum "$SELECT_SCRIPT" | awk '{print $1}')" == "$SELECT_SCRIPT_SHA256" ]] || { printf 'q36 group selection script differs\n' >&2; exit 2; }
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" && ! -e "$REPORT" && ! -L "$REPORT" ]] || { printf 'q36 group selection output exists\n' >&2; exit 2; }

IFS=: read -r -a first <<< "$FIRST_CANDIDATES"
IFS=: read -r -a second <<< "$SECOND_CANDIDATES"
args=(--first-label "$FIRST_LABEL" --second-label "$SECOND_LABEL" --split "$SPLIT" --output "$OUTPUT" --report "$REPORT")
for path in "${first[@]}"; do args+=(--first-candidates "$path"); done
for path in "${second[@]}"; do args+=(--second-candidates "$path"); done
export PYTHONPATH="$(dirname "$SELECT_SCRIPT")"
"$PYTHON" -P -s -B "$SELECT_SCRIPT" "${args[@]}"
chmod a-w "$OUTPUT" "$REPORT"
