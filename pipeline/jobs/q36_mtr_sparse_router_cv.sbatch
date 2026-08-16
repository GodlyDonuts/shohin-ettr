#!/bin/bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36-mtr-sparse-cv-%j.out
#SBATCH --error=logs/q36-mtr-sparse-cv-%j.err

set -euo pipefail

for name in PYTHON CV_SCRIPT CV_SCRIPT_SHA256 DEVELOPMENT_SOURCE CURRENT_CANDIDATES OWNER71_CANDIDATES OWNER8_CANDIDATES CURRENT_SCORES OWNER71_SCORES OWNER8_SCORES OUTPUT; do
  [[ -n "${!name:-}" ]] || { printf 'q36 sparse CV missing %s\n' "$name" >&2; exit 2; }
done
[[ -x "$PYTHON" && -f "$CV_SCRIPT" && ! -L "$CV_SCRIPT" ]] || { printf 'q36 sparse CV executable differs\n' >&2; exit 2; }
[[ "$(sha256sum "$CV_SCRIPT" | awk '{print $1}')" == "$CV_SCRIPT_SHA256" ]] || { printf 'q36 sparse CV script differs\n' >&2; exit 2; }
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || { printf 'q36 sparse CV output exists\n' >&2; exit 2; }

args=(--development-source "$DEVELOPMENT_SOURCE" --output "$OUTPUT")
for owner in current owner71 owner8; do
  candidate_name="${owner^^}_CANDIDATES"
  score_name="${owner^^}_SCORES"
  IFS=: read -r -a candidates <<< "${!candidate_name}"
  IFS=: read -r -a scores <<< "${!score_name}"
  [[ ${#candidates[@]} -eq 16 && ${#scores[@]} -eq 16 ]] || { printf 'q36 sparse CV shard geometry differs\n' >&2; exit 2; }
  for path in "${candidates[@]}"; do args+=(--"$owner"-candidates "$path"); done
  for path in "${scores[@]}"; do args+=(--"$owner"-scores "$path"); done
done

export PYTHONPATH="$(dirname "$CV_SCRIPT")"
"$PYTHON" -P -s -B "$CV_SCRIPT" "${args[@]}"
chmod a-w "$OUTPUT"
