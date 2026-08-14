#!/bin/bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36-mtr-synthesis-consensus-%j.out
#SBATCH --error=logs/q36-mtr-synthesis-consensus-%j.err

set -euo pipefail

for name in PYTHON CONSENSUS_SCRIPT CONSENSUS_SCRIPT_SHA256 CURRENT_CANDIDATES OWNER71_CANDIDATES OWNER8_CANDIDATES SYNTHESIS_CANDIDATES OUTPUT DECISIONS REPORT; do
  [[ -n "${!name:-}" ]] || { printf 'q36 synthesis consensus missing %s\n' "$name" >&2; exit 2; }
done
[[ -x "$PYTHON" && -f "$CONSENSUS_SCRIPT" && ! -L "$CONSENSUS_SCRIPT" ]] || { printf 'q36 synthesis consensus executable differs\n' >&2; exit 2; }
[[ "$(sha256sum "$CONSENSUS_SCRIPT" | awk '{print $1}')" == "$CONSENSUS_SCRIPT_SHA256" ]] || { printf 'q36 synthesis consensus script differs\n' >&2; exit 2; }
for path in "$OUTPUT" "$DECISIONS" "$REPORT"; do [[ ! -e "$path" && ! -L "$path" ]] || { printf 'q36 synthesis consensus output exists\n' >&2; exit 2; }; done
args=(--output "$OUTPUT" --decisions "$DECISIONS" --report "$REPORT")
for name in current owner71 owner8 synthesis; do
  variable="${name^^}_CANDIDATES"; IFS=: read -r -a paths <<< "${!variable}"
  [[ ${#paths[@]} -eq 16 ]] || { printf 'q36 synthesis consensus shard geometry differs\n' >&2; exit 2; }
  for path in "${paths[@]}"; do args+=(--"$name"-candidates "$path"); done
done
export PYTHONPATH="$(dirname "$CONSENSUS_SCRIPT"):$RUNTIME/train"
"$PYTHON" -P -s -B "$CONSENSUS_SCRIPT" "${args[@]}"
chmod a-w "$OUTPUT" "$DECISIONS" "$REPORT"
