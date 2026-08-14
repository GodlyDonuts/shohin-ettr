#!/bin/bash
#SBATCH --job-name=q36-multi-owner-pairs
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_multi_owner_pairs_%j.out
#SBATCH --error=logs/q36_multi_owner_pairs_%j.err

set -euo pipefail
for variable in PYTHON RUNTIME BUILDER_SCRIPT BUILDER_SCRIPT_SHA256 TRAIN_SOURCE DEVELOPMENT_SOURCE CURRENT_CANDIDATES OWNER71_CANDIDATES OWNER8_CANDIDATES CURRENT_TRAIN_SCORES OWNER71_TRAIN_SCORES OWNER8_TRAIN_SCORES CURRENT_DEVELOPMENT_SCORES OWNER71_DEVELOPMENT_SCORES OWNER8_DEVELOPMENT_SCORES TRAINING_OUTPUT TRAINING_REPORT DEVELOPMENT_OUTPUT DEVELOPMENT_REPORT; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
[[ -f "$BUILDER_SCRIPT" && ! -L "$BUILDER_SCRIPT" ]] || { printf 'builder is absent or linked\n' >&2; exit 2; }
[[ "$(sha256sum -- "$BUILDER_SCRIPT" | awk '{print $1}')" == "$BUILDER_SCRIPT_SHA256" ]] || { printf 'builder bytes differ\n' >&2; exit 2; }
for path in "$TRAIN_SOURCE" "$DEVELOPMENT_SOURCE"; do
  [[ -f "$path" && ! -L "$path" ]] || { printf 'source is absent or linked: %s\n' "$path" >&2; exit 2; }
done
IFS=: read -r -a current_candidates <<< "$CURRENT_CANDIDATES"
IFS=: read -r -a owner71_candidates <<< "$OWNER71_CANDIDATES"
IFS=: read -r -a owner8_candidates <<< "$OWNER8_CANDIDATES"
IFS=: read -r -a current_train_scores <<< "$CURRENT_TRAIN_SCORES"
IFS=: read -r -a owner71_train_scores <<< "$OWNER71_TRAIN_SCORES"
IFS=: read -r -a owner8_train_scores <<< "$OWNER8_TRAIN_SCORES"
IFS=: read -r -a current_development_scores <<< "$CURRENT_DEVELOPMENT_SCORES"
IFS=: read -r -a owner71_development_scores <<< "$OWNER71_DEVELOPMENT_SCORES"
IFS=: read -r -a owner8_development_scores <<< "$OWNER8_DEVELOPMENT_SCORES"
for count in "${#current_candidates[@]}" "${#owner71_candidates[@]}" "${#owner8_candidates[@]}" "${#current_train_scores[@]}" "${#owner71_train_scores[@]}" "${#owner8_train_scores[@]}" "${#current_development_scores[@]}" "${#owner71_development_scores[@]}" "${#owner8_development_scores[@]}"; do
  [[ "$count" -eq 16 ]] || { printf 'multi-owner shard geometry differs\n' >&2; exit 2; }
done
for path in "${current_candidates[@]}" "${owner71_candidates[@]}" "${owner8_candidates[@]}" "${current_train_scores[@]}" "${owner71_train_scores[@]}" "${owner8_train_scores[@]}" "${current_development_scores[@]}" "${owner71_development_scores[@]}" "${owner8_development_scores[@]}"; do
  [[ -f "$path" && ! -L "$path" ]] || { printf 'multi-owner input is absent or linked: %s\n' "$path" >&2; exit 2; }
done
for path in "$TRAINING_OUTPUT" "$TRAINING_REPORT" "$DEVELOPMENT_OUTPUT" "$DEVELOPMENT_REPORT"; do
  [[ ! -e "$path" && ! -L "$path" ]] || { printf 'multi-owner output exists: %s\n' "$path" >&2; exit 2; }
done
mkdir -p "$(dirname "$TRAINING_OUTPUT")" "$(dirname "$DEVELOPMENT_OUTPUT")" logs
arguments=()
for path in "${current_candidates[@]}"; do arguments+=(--current-candidates "$path"); done
for path in "${owner71_candidates[@]}"; do arguments+=(--owner71-candidates "$path"); done
for path in "${owner8_candidates[@]}"; do arguments+=(--owner8-candidates "$path"); done
for path in "${current_train_scores[@]}"; do arguments+=(--current-train-score "$path"); done
for path in "${owner71_train_scores[@]}"; do arguments+=(--owner71-train-score "$path"); done
for path in "${owner8_train_scores[@]}"; do arguments+=(--owner8-train-score "$path"); done
for path in "${current_development_scores[@]}"; do arguments+=(--current-development-score "$path"); done
for path in "${owner71_development_scores[@]}"; do arguments+=(--owner71-development-score "$path"); done
for path in "${owner8_development_scores[@]}"; do arguments+=(--owner8-development-score "$path"); done
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$RUNTIME/pipeline:$RUNTIME/train"
"$PYTHON" -P -s -B "$BUILDER_SCRIPT" \
  --train-source "$TRAIN_SOURCE" --development-source "$DEVELOPMENT_SOURCE" \
  "${arguments[@]}" \
  --training-output "$TRAINING_OUTPUT" --training-report "$TRAINING_REPORT" \
  --development-output "$DEVELOPMENT_OUTPUT" --development-report "$DEVELOPMENT_REPORT" \
  --seed 2026080820
for path in "$TRAINING_OUTPUT" "$TRAINING_REPORT" "$DEVELOPMENT_OUTPUT" "$DEVELOPMENT_REPORT"; do
  [[ -f "$path" ]] || { printf 'multi-owner output is missing: %s\n' "$path" >&2; exit 2; }
  chmod a-w "$path"
done
