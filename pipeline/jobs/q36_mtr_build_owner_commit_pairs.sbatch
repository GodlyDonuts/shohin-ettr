#!/bin/bash
#SBATCH --job-name=q36-owner-commit-pairs
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_owner_commit_pairs_%j.out
#SBATCH --error=logs/q36_owner_commit_pairs_%j.err

set -euo pipefail
for variable in PYTHON RUNTIME BUILDER_SCRIPT BUILDER_SCRIPT_SHA256 TRAIN_SOURCE DEVELOPMENT_SOURCE FIRST_CANDIDATES SECOND_CANDIDATES FIRST_TRAIN_SCORES SECOND_TRAIN_SCORES FIRST_DEVELOPMENT_SCORES SECOND_DEVELOPMENT_SCORES TRAINING_OUTPUT TRAINING_REPORT DEVELOPMENT_OUTPUT DEVELOPMENT_REPORT; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
sha256_file() { sha256sum -- "$1" | awk '{print $1}'; }
[[ -f "$BUILDER_SCRIPT" && ! -L "$BUILDER_SCRIPT" ]] || { printf 'builder is absent or linked\n' >&2; exit 2; }
[[ "$(sha256_file "$BUILDER_SCRIPT")" == "$BUILDER_SCRIPT_SHA256" ]] || { printf 'builder bytes differ\n' >&2; exit 2; }
for path in "$TRAIN_SOURCE" "$DEVELOPMENT_SOURCE"; do
  [[ -f "$path" && ! -L "$path" ]] || { printf 'source is absent or linked: %s\n' "$path" >&2; exit 2; }
done
IFS=: read -r -a first_candidates <<< "$FIRST_CANDIDATES"
IFS=: read -r -a second_candidates <<< "$SECOND_CANDIDATES"
IFS=: read -r -a first_train_scores <<< "$FIRST_TRAIN_SCORES"
IFS=: read -r -a second_train_scores <<< "$SECOND_TRAIN_SCORES"
IFS=: read -r -a first_development_scores <<< "$FIRST_DEVELOPMENT_SCORES"
IFS=: read -r -a second_development_scores <<< "$SECOND_DEVELOPMENT_SCORES"
for count in "${#first_candidates[@]}" "${#second_candidates[@]}" "${#first_train_scores[@]}" "${#second_train_scores[@]}" "${#first_development_scores[@]}" "${#second_development_scores[@]}"; do
  [[ "$count" -eq 16 ]] || { printf 'owner-commit shard geometry differs\n' >&2; exit 2; }
done
for path in "${first_candidates[@]}" "${second_candidates[@]}" "${first_train_scores[@]}" "${second_train_scores[@]}" "${first_development_scores[@]}" "${second_development_scores[@]}"; do
  [[ -f "$path" && ! -L "$path" ]] || { printf 'owner-commit input is absent or linked: %s\n' "$path" >&2; exit 2; }
done
for path in "$TRAINING_OUTPUT" "$TRAINING_REPORT" "$DEVELOPMENT_OUTPUT" "$DEVELOPMENT_REPORT"; do
  [[ ! -e "$path" && ! -L "$path" ]] || { printf 'owner-commit output exists: %s\n' "$path" >&2; exit 2; }
done
mkdir -p "$(dirname "$TRAINING_OUTPUT")" "$(dirname "$DEVELOPMENT_OUTPUT")" logs
arguments=()
for path in "${first_candidates[@]}"; do arguments+=(--first-candidates "$path"); done
for path in "${second_candidates[@]}"; do arguments+=(--second-candidates "$path"); done
for path in "${first_train_scores[@]}"; do arguments+=(--first-train-score "$path"); done
for path in "${second_train_scores[@]}"; do arguments+=(--second-train-score "$path"); done
for path in "${first_development_scores[@]}"; do arguments+=(--first-development-score "$path"); done
for path in "${second_development_scores[@]}"; do arguments+=(--second-development-score "$path"); done
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$RUNTIME/pipeline:$RUNTIME/train"
"$PYTHON" -P -s -B "$BUILDER_SCRIPT" \
  --train-source "$TRAIN_SOURCE" \
  --development-source "$DEVELOPMENT_SOURCE" \
  "${arguments[@]}" \
  --training-output "$TRAINING_OUTPUT" \
  --training-report "$TRAINING_REPORT" \
  --development-output "$DEVELOPMENT_OUTPUT" \
  --development-report "$DEVELOPMENT_REPORT" \
  --seed 2026080820
for path in "$TRAINING_OUTPUT" "$TRAINING_REPORT" "$DEVELOPMENT_OUTPUT" "$DEVELOPMENT_REPORT"; do
  [[ -f "$path" ]] || { printf 'owner-commit output is missing: %s\n' "$path" >&2; exit 2; }
  chmod a-w "$path"
done
