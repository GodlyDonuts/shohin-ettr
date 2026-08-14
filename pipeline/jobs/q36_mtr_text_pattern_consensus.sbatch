#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --no-requeue

set -euo pipefail

for name in \
  PYTHON SELECTOR SOURCE OUTPUT_ROOT DEPENDENCY_PATHS \
  HIERARCHY_ROOT INTERPOLATION_ROOT DIRECT_ROOT OFFSET_ONE_ROOT \
  LEVEL_TWO_ROOT CHALLENGER_ROOT; do
  [[ -n "${!name:-}" ]] || { echo "q36-text-pattern: $name is required" >&2; exit 2; }
done

[[ -x "$PYTHON" && -f "$SELECTOR" && -f "$SOURCE" && -d "$OUTPUT_ROOT" ]] || {
  echo "q36-text-pattern: executable/input/output projection differs" >&2
  exit 2
}
[[ ! -e "$OUTPUT_ROOT/candidates.jsonl" && ! -e "$OUTPUT_ROOT/report.json" ]] || {
  echo "q36-text-pattern: output already exists" >&2
  exit 2
}

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$DEPENDENCY_PATHS"

arms=(hierarchy interpolation direct offset_one level_two challenger)
roots=(
  "$HIERARCHY_ROOT"
  "$INTERPOLATION_ROOT"
  "$DIRECT_ROOT"
  "$OFFSET_ONE_ROOT"
  "$LEVEL_TWO_ROOT"
  "$CHALLENGER_ROOT"
)
args=()
for index in "${!arms[@]}"; do
  option="${arms[$index]//_/-}"
  root="${roots[$index]}"
  [[ -d "$root" ]] || { echo "q36-text-pattern: arm root missing" >&2; exit 2; }
  for shard in $(seq -w 0 15); do
    candidates="$root/shard_$shard/candidates.jsonl"
    score="$root/shard_$shard/development_score.json"
    [[ -f "$candidates" && -f "$score" ]] || {
      echo "q36-text-pattern: shard input missing" >&2
      exit 2
    }
    args+=("--$option" "$candidates" "--$option-score" "$score")
  done
done

"$PYTHON" -P -s -B "$SELECTOR" \
  "${args[@]}" \
  --source "$SOURCE" \
  --output "$OUTPUT_ROOT/candidates.jsonl" \
  --report "$OUTPUT_ROOT/report.json"

chmod a-w "$OUTPUT_ROOT/candidates.jsonl" "$OUTPUT_ROOT/report.json"
