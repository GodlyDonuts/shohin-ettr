#!/bin/bash
#SBATCH -J eal2-confirm
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:20:00
#SBATCH -o /lustre/fs1/home/%u/shohin/logs/diverge_eal2_confirmation_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable EAL2 confirmation code root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set EAL2 confirmation source commit}
PRIMARY_ROOT=${PRIMARY_ROOT:?set frozen passing EAL2 development root}
DEV_RESULT_SHA256=${DEV_RESULT_SHA256:?set passing development result hash}
CHECKPOINT_SHA256=${CHECKPOINT_SHA256:?set frozen EAL2 checkpoint hash}
TRAINING_REPORT_SHA256=${TRAINING_REPORT_SHA256:?set frozen EAL2 training report hash}
PUBLIC_DATA=${PUBLIC_DATA:?set immutable confirmation public board}
PUBLIC_SHA256=${PUBLIC_SHA256:?set confirmation public hash}
ASSESSOR_DATA=${ASSESSOR_DATA:?set immutable confirmation assessor board}
ASSESSOR_SHA256=${ASSESSOR_SHA256:?set confirmation assessor hash}
OUTPUT=${OUTPUT:?set isolated EAL2 confirmation result}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

CHECKPOINT=$PRIMARY_ROOT/training/checkpoint.pt
TRAINING_REPORT=$PRIMARY_ROOT/training/report.json
DEV_RESULT=$PRIMARY_ROOT/development.json
export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs" "$(dirname "$OUTPUT")"
(cd "$CODE_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$DEV_RESULT" | cut -d' ' -f1)" = "$DEV_RESULT_SHA256"
test "$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$DEV_RESULT")" = pass
test "$(sha256sum "$CHECKPOINT" | cut -d' ' -f1)" = "$CHECKPOINT_SHA256"
test "$(sha256sum "$TRAINING_REPORT" | cut -d' ' -f1)" = "$TRAINING_REPORT_SHA256"
test "$(sha256sum "$PUBLIC_DATA" | cut -d' ' -f1)" = "$PUBLIC_SHA256"
test "$(sha256sum "$ASSESSOR_DATA" | cut -d' ' -f1)" = "$ASSESSOR_SHA256"
test ! -e "$OUTPUT"
trap 'test ! -e "$OUTPUT" || chmod a-w "$OUTPUT"' EXIT
nvidia-smi -L
"$PYTHON" "$CODE_ROOT/train/test_diverge_eal2_runtime.py"
"$PYTHON" "$CODE_ROOT/train/eval_diverge_eal2.py" \
  --checkpoint "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --training-report "$TRAINING_REPORT" \
  --training-report-sha256 "$TRAINING_REPORT_SHA256" \
  --public-data "$PUBLIC_DATA" --public-data-sha256 "$PUBLIC_SHA256" \
  --assessor-data "$ASSESSOR_DATA" --assessor-data-sha256 "$ASSESSOR_SHA256" \
  --output "$OUTPUT" --source-commit "$SOURCE_COMMIT" --batch-size 512
test "$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$OUTPUT")" = pass
echo "[diverge-eal2-confirmation] completed"
