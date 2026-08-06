#!/bin/bash
#SBATCH -J eval-fta1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/eval_diverge_fta1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
SOURCE_COMMIT=${SOURCE_COMMIT:?set immutable SOURCE_COMMIT}
TRAIN_OUTPUT=${TRAIN_OUTPUT:?set completed FTA1 training output}
OUTPUT=${OUTPUT:?set isolated FTA1 evaluation JSON}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
DATA=$BASE/artifacts/reasoning/diverge_crp1/evaluation_seed2026080604.jsonl
DATA_SHA256=db0bde0c22afe3d25f4f1f578249bf67156f4115d38a094c07f1ea36f6be6849

export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs" "$(dirname "$OUTPUT")"
cd "$BASE"

readarray -t RECEIPT < <("$PYTHON" - "$TRAIN_OUTPUT/report.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("schema") != "shohin-diverge-fta1-training-report-v1":
    raise SystemExit("FTA1 training report schema differs")
print(report["checkpoint"])
print(report["checkpoint_sha256"])
PY
)
CHECKPOINT=${RECEIPT[0]}
CHECKPOINT_SHA256=${RECEIPT[1]}
test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$CHECKPOINT_SHA256"
test "$(sha256sum "$DATA" | awk '{print $1}')" = "$DATA_SHA256"
test ! -e "$OUTPUT" || { echo "refusing existing output: $OUTPUT" >&2; exit 2; }
trap 'test ! -e "$OUTPUT" || chmod a-w "$OUTPUT"' EXIT

echo "[eval-fta1] node=${SLURMD_NODENAME:-?} code=$CODE_ROOT commit=$SOURCE_COMMIT"
nvidia-smi -L
"$PYTHON" "$CODE_ROOT/train/eval_diverge_fta1.py" \
  --checkpoint "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --expected-update 1600 --data "$DATA" --data-sha256 "$DATA_SHA256" \
  --output "$OUTPUT" --batch-size 1024

echo "[eval-fta1] completed output=$OUTPUT"
