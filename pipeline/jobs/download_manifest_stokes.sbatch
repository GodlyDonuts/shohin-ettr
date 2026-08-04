#!/bin/bash
#SBATCH --job-name=pinned-download
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pinned_download_%A_%a.out
#SBATCH --error=logs/pinned_download_%A_%a.err

set -euo pipefail

: "${MANIFEST:?MANIFEST is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
INDEX=${SLURM_ARRAY_TASK_ID:?array task id is required}

test -s "$MANIFEST"
LINE=$(sed -n "$((INDEX + 1))p" "$MANIFEST")
test -n "$LINE"
RELATIVE_PATH=${LINE%%$'\t'*}
URL=${LINE#*$'\t'}
test -n "$RELATIVE_PATH"
test "$URL" != "$LINE"
[[ "$URL" == https://* ]]
[[ "$RELATIVE_PATH" != /* ]]
[[ "/$RELATIVE_PATH/" != *"/../"* ]]

DESTINATION="$OUTPUT_ROOT/$RELATIVE_PATH"
PARTIAL="$DESTINATION.part"
HASH_FILE="$DESTINATION.sha256"
test ! -e "$DESTINATION"
test ! -e "$PARTIAL"
test ! -e "$HASH_FILE"
mkdir -p logs "$(dirname "$DESTINATION")"

curl --fail --location --retry 20 --retry-delay 15 \
  --connect-timeout 30 --output "$PARTIAL" "$URL"
test -s "$PARTIAL"
HASH=$(sha256sum "$PARTIAL" | cut -d' ' -f1)
mv "$PARTIAL" "$DESTINATION"
printf '%s  %s\n' "$HASH" "$RELATIVE_PATH" > "$HASH_FILE"
