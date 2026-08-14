#!/bin/bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36-mtr-sparse-router-%j.out
#SBATCH --error=logs/q36-mtr-sparse-router-%j.err

set -euo pipefail

for name in PYTHON ROUTER_SCRIPT ROUTER_SCRIPT_SHA256 TRAINING_ROWS DEVELOPMENT_ROWS CURRENT_CANDIDATES OWNER71_CANDIDATES OWNER8_CANDIDATES MODEL_OUTPUT OUTPUT SELECTIONS REPORT; do
  [[ -n "${!name:-}" ]] || { printf 'q36 sparse router missing %s\n' "$name" >&2; exit 2; }
done
[[ -x "$PYTHON" && -f "$ROUTER_SCRIPT" && ! -L "$ROUTER_SCRIPT" ]] || { printf 'q36 sparse router executable differs\n' >&2; exit 2; }
[[ "$(sha256sum "$ROUTER_SCRIPT" | awk '{print $1}')" == "$ROUTER_SCRIPT_SHA256" ]] || { printf 'q36 sparse router script differs\n' >&2; exit 2; }
for path in "$MODEL_OUTPUT" "$OUTPUT" "$SELECTIONS" "$REPORT"; do
  [[ ! -e "$path" && ! -L "$path" ]] || { printf 'q36 sparse router output exists: %s\n' "$path" >&2; exit 2; }
done

IFS=: read -r -a current <<< "$CURRENT_CANDIDATES"
IFS=: read -r -a owner71 <<< "$OWNER71_CANDIDATES"
IFS=: read -r -a owner8 <<< "$OWNER8_CANDIDATES"
[[ ${#current[@]} -eq 16 && ${#owner71[@]} -eq 16 && ${#owner8[@]} -eq 16 ]] || { printf 'q36 sparse router shard geometry differs\n' >&2; exit 2; }

args=(
  --training-rows "$TRAINING_ROWS"
  --development-rows "$DEVELOPMENT_ROWS"
  --model-output "$MODEL_OUTPUT"
  --output "$OUTPUT"
  --selections "$SELECTIONS"
  --report "$REPORT"
)
for path in "${current[@]}"; do args+=(--current-candidates "$path"); done
for path in "${owner71[@]}"; do args+=(--owner71-candidates "$path"); done
for path in "${owner8[@]}"; do args+=(--owner8-candidates "$path"); done

"$PYTHON" -P -s -B "$ROUTER_SCRIPT" "${args[@]}"
chmod a-w "$MODEL_OUTPUT" "$OUTPUT" "$SELECTIONS" "$REPORT"
