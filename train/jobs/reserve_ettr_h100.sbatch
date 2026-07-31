#!/bin/bash
#SBATCH --job-name=shohin-ettr-h100
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc32,evc33,evc43
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/reserve-ettr-h100-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/reserve-ettr-h100-%j.err

set -euo pipefail

mkdir -p /lustre/fs1/home/sa305415/shohin/logs
printf 'reservation job=%s node=%s started=%s\n' \
  "${SLURM_JOB_ID:?}" "${SLURMD_NODENAME:?}" "$(date --iso-8601=seconds)"

# Training and evaluation enter the allocation through audited srun steps.
while :; do
  sleep 3600
done
