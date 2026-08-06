#!/bin/bash
#SBATCH -J crp1-board
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/diverge_crp1_board_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
SOURCE_COMMIT=${SOURCE_COMMIT:?set immutable SOURCE_COMMIT}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL=$BASE/artifacts/external/smollm3-3b-a07cc9a
TRAIN=${TRAIN:-$BASE/artifacts/reasoning/diverge_crp1/train_seed2026080604.jsonl}
DEVELOPMENT=${DEVELOPMENT:-$BASE/artifacts/reasoning/diverge_crp1/development_seed2026080604.jsonl}
EVALUATION=${EVALUATION:-$BASE/artifacts/reasoning/diverge_crp1/evaluation_seed2026080604.jsonl}
REPORT=${REPORT:-$BASE/artifacts/reasoning/diverge_crp1/board_seed2026080604.report.json}

export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" logs
cd "$BASE"

test "$(sha256sum "$MODEL/config.json" | awk '{print $1}')" = \
  c72b1031274ff4626e434d0019e88e95a767460135db9ee492eb80652b786af1
test "$(sha256sum "$MODEL/tokenizer.json" | awk '{print $1}')" = \
  7b6a500b662a34eb3f0374db856ba4ad7de4c81040571d78dc0d357238930005
for path in "$TRAIN" "$DEVELOPMENT" "$EVALUATION" "$REPORT"; do test ! -e "$path"; done
mkdir -p "$(dirname "$TRAIN")"
trap 'for path in "$TRAIN" "$DEVELOPMENT" "$EVALUATION" "$REPORT"; do test ! -e "$path" || chmod a-w "$path"; done' EXIT

echo "[crp1-board] node=${SLURMD_NODENAME:-?} code=$CODE_ROOT commit=$SOURCE_COMMIT"
"$PYTHON" -m py_compile \
  "$CODE_ROOT/pipeline/build_diverge_crp1_board.py" \
  "$CODE_ROOT/train/diverge_crp1_data.py"
"$PYTHON" -c 'import build_diverge_crp1_board, diverge_crp1_data'
"$PYTHON" "$CODE_ROOT/pipeline/build_diverge_crp1_board.py" \
  --model-root "$MODEL" \
  --train-output "$TRAIN" \
  --development-output "$DEVELOPMENT" \
  --evaluation-output "$EVALUATION" \
  --report "$REPORT" \
  --train-per-family 1600 \
  --development-per-family 160 \
  --evaluation-per-family 160 \
  --train-min-depth 4 --train-max-depth 6 \
  --eval-min-depth 7 --eval-max-depth 9 \
  --max-sequence-length 4096 \
  --workspace-slots 6 \
  --seed 2026080604

sha256sum "$TRAIN" "$DEVELOPMENT" "$EVALUATION" "$REPORT"
echo "[crp1-board] completed"
