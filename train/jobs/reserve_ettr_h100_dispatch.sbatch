#!/bin/bash
#SBATCH --job-name=shohin-ettr-dispatch
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc32,evc33,evc43
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/reserve-ettr-dispatch-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/reserve-ettr-dispatch-%j.err

set -euo pipefail
umask 077

ROOT=/lustre/fs1/home/sa305415/shohin
CONTROL_ROOT="$ROOT/control/ettr-h100-dispatch-v1"
job=${SLURM_JOB_ID:?}
node=${SLURMD_NODENAME:?}
ready="$CONTROL_ROOT/ready/$job"
command="$CONTROL_ROOT/commands/$job.sh"
command_sha="$CONTROL_ROOT/commands/$job.sha256"
status="$CONTROL_ROOT/status/$job"

mkdir -p "$ROOT/logs"
mkdir -p \
  "$CONTROL_ROOT/ready" \
  "$CONTROL_ROOT/commands" \
  "$CONTROL_ROOT/status"
for path in "$ready" "$command" "$command_sha" "$status"; do
  if [[ -e "$path" || -L "$path" ]]; then
    echo "dispatch control path already exists: $path" >&2
    exit 2
  fi
done

ready_tmp=$(mktemp "$CONTROL_ROOT/ready/.${job}.XXXXXX")
printf 'job=%s\nnode=%s\ngpus=%s\nstarted=%s\n' \
  "$job" "$node" "${SLURM_GPUS_ON_NODE:?}" "$(date --iso-8601=seconds)" \
  >"$ready_tmp"
chmod 400 "$ready_tmp"
mv -n "$ready_tmp" "$ready"
test -f "$ready"

while [[ ! -f "$command" || ! -f "$command_sha" ]]; do
  sleep 2
done
if [[ -L "$command" || -L "$command_sha" \
  || "$(stat -c %U "$command")" != "$USER" \
  || "$(stat -c %a "$command")" != 400 \
  || "$(stat -c %h "$command")" != 1 ]]; then
  echo "dispatch command custody differs" >&2
  exit 2
fi
read -r expected extra <"$command_sha"
if [[ ! "$expected" =~ ^[0-9a-f]{64}$ || -n "${extra:-}" ]]; then
  echo "dispatch command digest differs" >&2
  exit 2
fi

exec {command_fd}<"$command"
actual=$(sha256sum "/proc/self/fd/$command_fd" | awk '{print $1}')
if [[ "$actual" != "$expected" ]]; then
  echo "dispatch command content differs" >&2
  exit 2
fi

set +e
bash "/proc/self/fd/$command_fd"
result=$?
set -e
status_tmp=$(mktemp "$CONTROL_ROOT/status/.${job}.XXXXXX")
printf 'job=%s\nnode=%s\nexit_code=%s\nfinished=%s\ncommand_sha256=%s\n' \
  "$job" "$node" "$result" "$(date --iso-8601=seconds)" "$actual" \
  >"$status_tmp"
chmod 400 "$status_tmp"
mv -n "$status_tmp" "$status"
test -f "$status"
exit "$result"
