#!/bin/bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36-mtr-semantic-compare-%j.out
#SBATCH --error=logs/q36-mtr-semantic-compare-%j.err

set -euo pipefail

for name in PYTHON COMPARE_SCRIPT COMPARE_SCRIPT_SHA256 GROUPS OUTPUT; do
  [[ -n "${!name:-}" ]] || { printf 'q36 semantic comparison missing %s\n' "$name" >&2; exit 2; }
done
[[ -x "$PYTHON" && -f "$COMPARE_SCRIPT" && ! -L "$COMPARE_SCRIPT" ]] || { printf 'q36 semantic comparison executable differs\n' >&2; exit 2; }
[[ "$(sha256sum "$COMPARE_SCRIPT" | awk '{print $1}')" == "$COMPARE_SCRIPT_SHA256" ]] || { printf 'q36 semantic comparison script differs\n' >&2; exit 2; }
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || { printf 'q36 semantic comparison output exists\n' >&2; exit 2; }

IFS=';' read -r -a groups <<< "$GROUPS"
[[ ${#groups[@]} -ge 2 ]] || { printf 'q36 semantic comparison groups differ\n' >&2; exit 2; }
args=()
for group in "${groups[@]}"; do
  [[ "$group" == *:* ]] || { printf 'q36 semantic comparison group differs\n' >&2; exit 2; }
  args+=(--group "$group")
done

"$PYTHON" -P -s -B "$COMPARE_SCRIPT" "${args[@]}" --output "$OUTPUT"
chmod a-w "$OUTPUT"
