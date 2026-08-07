#!/bin/bash
#SBATCH -J eal1-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:20:00
#SBATCH -o /lustre/fs1/home/%u/shohin/logs/diverge_eal1_eval_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable patched EAL1 code root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set EAL1 reporting-patch commit}
DATA_ROOT=${DATA_ROOT:?set immutable EAL1 data root}
PRIMARY_ROOT=${PRIMARY_ROOT:?set frozen EAL1 primary root}
OUTPUT=${OUTPUT:?set isolated EAL1 replay report}
CHECKPOINT_SHA256=${CHECKPOINT_SHA256:?set frozen EAL1 checkpoint hash}
TRAINING_REPORT_SHA256=${TRAINING_REPORT_SHA256:?set frozen training report hash}
PUBLIC_SHA256=${PUBLIC_SHA256:?set EAL1 public development hash}
ASSESSOR_SHA256=${ASSESSOR_SHA256:?set EAL1 assessor development hash}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

CHECKPOINT=$PRIMARY_ROOT/training/checkpoint.pt
TRAINING_REPORT=$PRIMARY_ROOT/training/report.json
PUBLIC_DATA=$DATA_ROOT/development_public.jsonl
ASSESSOR_DATA=$DATA_ROOT/development_assessor.jsonl
export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs" "$(dirname "$OUTPUT")"
(cd "$CODE_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$CHECKPOINT" | cut -d' ' -f1)" = "$CHECKPOINT_SHA256"
test "$(sha256sum "$TRAINING_REPORT" | cut -d' ' -f1)" = "$TRAINING_REPORT_SHA256"
test "$(sha256sum "$PUBLIC_DATA" | cut -d' ' -f1)" = "$PUBLIC_SHA256"
test "$(sha256sum "$ASSESSOR_DATA" | cut -d' ' -f1)" = "$ASSESSOR_SHA256"
test ! -e "$OUTPUT"
trap 'test ! -e "$OUTPUT" || chmod a-w "$OUTPUT"' EXIT
nvidia-smi -L
"$PYTHON" "$CODE_ROOT/train/eval_diverge_eal1.py" \
  --checkpoint "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --training-report "$TRAINING_REPORT" \
  --training-report-sha256 "$TRAINING_REPORT_SHA256" \
  --public-data "$PUBLIC_DATA" --public-data-sha256 "$PUBLIC_SHA256" \
  --assessor-data "$ASSESSOR_DATA" --assessor-data-sha256 "$ASSESSOR_SHA256" \
  --output "$OUTPUT" --source-commit "$SOURCE_COMMIT" --batch-size 512
test "$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$OUTPUT")" = fail
echo "[diverge-eal1-eval] preserved frozen failure"
