#!/bin/bash
#SBATCH --job-name=q36-control-forest
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

set -euo pipefail

for variable in PYTHON RUNTIME SKLEARN_RUNTIME SOURCE_ROOT SCRIPT SCRIPT_SHA256 RUN OUTPUT_ROOT; do
  [[ -n "$(printenv "$variable")" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
[[ "$SCRIPT_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
[[ "$(sha256sum "$SCRIPT" | cut -d' ' -f1)" == "$SCRIPT_SHA256" ]] || exit 2
[[ -d "$SOURCE_ROOT" && -d "$SKLEARN_RUNTIME" && -d "$OUTPUT_ROOT" ]] || exit 2
[[ ! -e "$OUTPUT_ROOT/candidates.jsonl" && ! -e "$OUTPUT_ROOT/report.json" ]] || exit 2

export PATH=/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=$SKLEARN_RUNTIME:$SOURCE_ROOT:$RUNTIME/train:$RUNTIME/pipeline

ep=$RUN/engineering_preview
arms=(hierarchy interpolation direct offset_one level_two challenger)
roots=(
  hierarchical_synthesis_468b30c1_r2
  checkpoint_interpolation_e053de52_r1/w010/drafts
  trajectory_synthesis_8ca1b86_r2
  trajectory_synthesis_5d6a47d1_offset1
  trajectory_synthesis_5d6a47d1_level2
  incumbent_challenger_d66fc88e_r1
)
args=()
for index in 0 1 2 3 4 5; do
  arm=${arms[$index]}
  option=${arm//_/-}
  root=$ep/${roots[$index]}
  for shard in $(seq -w 0 15); do
    candidate=$root/shard_$shard/candidates.jsonl
    score=$root/shard_$shard/development_score.json
    [[ -f "$candidate" && -f "$score" ]] || exit 2
    args+=(--$option "$candidate" --$option-score "$score")
  done
done
for shard in $(seq -w 00 07); do
  candidate=$RUN/evaluations/development_unchanged/shard_$shard/candidates.jsonl
  score=$ep/matched_controls_4cb6e9d/scores/unchanged/shard_$shard.json
  [[ -f "$candidate" && -f "$score" ]] || exit 2
  args+=(--control-candidates "$candidate" --control-score "$score")
done

forest=$ep/nested_forest_consensus_d0255762_r1/output
"$PYTHON" "$SCRIPT" "${args[@]}" \
  --forest-candidates "$forest/candidates.jsonl" \
  --forest-report "$forest/report.json" \
  --output "$OUTPUT_ROOT/candidates.jsonl" \
  --report "$OUTPUT_ROOT/report.json"
chmod a-w "$OUTPUT_ROOT/candidates.jsonl" "$OUTPUT_ROOT/report.json"
