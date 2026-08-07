#!/bin/bash
#SBATCH -J diverge-eal2
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/%u/shohin/logs/diverge_eal2_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable EAL2 code root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set frozen EAL2 source commit}
DATA_ROOT=${DATA_ROOT:?set immutable EAL2 data root}
TRAINING_SHA256=${TRAINING_SHA256:?set shared training hash}
PUBLIC_SHA256=${PUBLIC_SHA256:?set shared public development hash}
ASSESSOR_SHA256=${ASSESSOR_SHA256:?set shared assessor development hash}
OUTPUT_ROOT=${OUTPUT_ROOT:?set isolated EAL2 output root}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

TRAINING_DATA=$DATA_ROOT/training.jsonl
PUBLIC_DATA=$DATA_ROOT/development_public.jsonl
ASSESSOR_DATA=$DATA_ROOT/development_assessor.jsonl
TRAINING_OUTPUT=$OUTPUT_ROOT/training
DEVELOPMENT_OUTPUT=$OUTPUT_ROOT/development.json
export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs"
(cd "$CODE_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$TRAINING_DATA" | cut -d' ' -f1)" = "$TRAINING_SHA256"
test "$(sha256sum "$PUBLIC_DATA" | cut -d' ' -f1)" = "$PUBLIC_SHA256"
test "$(sha256sum "$ASSESSOR_DATA" | cut -d' ' -f1)" = "$ASSESSOR_SHA256"
test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"
trap 'chmod -R a-w "$OUTPUT_ROOT" 2>/dev/null || true' EXIT

echo "[diverge-eal2] node=${SLURMD_NODENAME:-?} code=$CODE_ROOT commit=$SOURCE_COMMIT"
nvidia-smi -L
"$PYTHON" -m py_compile \
  "$CODE_ROOT/train/diverge_eal1_data.py" \
  "$CODE_ROOT/train/diverge_eal1_runtime.py" \
  "$CODE_ROOT/train/diverge_eal2_data.py" \
  "$CODE_ROOT/train/diverge_eal2_runtime.py" \
  "$CODE_ROOT/train/train_diverge_eal2.py" \
  "$CODE_ROOT/train/eval_diverge_eal2.py"
"$PYTHON" "$CODE_ROOT/train/test_diverge_eal1_runtime.py"
"$PYTHON" "$CODE_ROOT/train/test_diverge_eal2_data.py"
"$PYTHON" "$CODE_ROOT/train/test_diverge_eal2_runtime.py"
"$PYTHON" "$CODE_ROOT/train/train_diverge_eal2.py" \
  --data "$TRAINING_DATA" --data-sha256 "$TRAINING_SHA256" \
  --output "$TRAINING_OUTPUT" --source-commit "$SOURCE_COMMIT" \
  --updates 1000 --batch-size 256 --evaluation-batch-size 512 \
  --learning-rate 0.003 --seed 2026080761 --log-interval 25

CHECKPOINT=$TRAINING_OUTPUT/checkpoint.pt
TRAINING_REPORT=$TRAINING_OUTPUT/report.json
CHECKPOINT_SHA256=$(sha256sum "$CHECKPOINT" | cut -d' ' -f1)
TRAINING_REPORT_SHA256=$(sha256sum "$TRAINING_REPORT" | cut -d' ' -f1)
"$PYTHON" "$CODE_ROOT/train/eval_diverge_eal2.py" \
  --checkpoint "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --training-report "$TRAINING_REPORT" \
  --training-report-sha256 "$TRAINING_REPORT_SHA256" \
  --public-data "$PUBLIC_DATA" --public-data-sha256 "$PUBLIC_SHA256" \
  --assessor-data "$ASSESSOR_DATA" --assessor-data-sha256 "$ASSESSOR_SHA256" \
  --output "$DEVELOPMENT_OUTPUT" --source-commit "$SOURCE_COMMIT" \
  --batch-size 512
test "$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$DEVELOPMENT_OUTPUT")" = pass
echo "[diverge-eal2] completed"
