#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/ettr-only-joint-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/ettr-only-joint-%j.err

set -euo pipefail

COMMAND_FILE=${COMMAND_FILE:?set immutable command file}
COMMAND_SHA256=${COMMAND_SHA256:?set immutable command SHA-256}

[[ "$COMMAND_FILE" == /* && "$COMMAND_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ -f "$COMMAND_FILE" && ! -L "$COMMAND_FILE" ]]

snapshot=$(mktemp)
environment=$(mktemp)
cleanup() {
  rm -f "$snapshot" "$environment"
}
trap cleanup EXIT

cp -- "$COMMAND_FILE" "$snapshot"
printf '%s  %s\n' "$COMMAND_SHA256" "$snapshot" | sha256sum -c -

expected_launcher='exec "$CODE_ROOT/train/jobs/run_reserved_ettr_v3_pilot.sh"'
[[ "$(tail -n 1 "$snapshot")" == "$expected_launcher" ]]
head -n -1 "$snapshot" > "$environment"

# shellcheck disable=SC1090
source "$environment"
rm -f "$snapshot" "$environment"
trap - EXIT

[[ -f "$CODE_ROOT/train/jobs/run_reserved_ettr_v3_pilot.sh" ]]
exec bash "$CODE_ROOT/train/jobs/run_reserved_ettr_v3_pilot.sh"
