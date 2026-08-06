#!/bin/bash
#SBATCH -J vcr1-board
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/diverge_vcr1_board_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
SOURCE_COMMIT=${SOURCE_COMMIT:?set immutable SOURCE_COMMIT}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL=$BASE/artifacts/external/smollm3-3b-a07cc9a
SOURCE=$BASE/artifacts/product_reasoning/data/rest_rollout_preferences_freshv3_k4_p2_r1.jsonl
TRAIN=${TRAIN:-$BASE/artifacts/reasoning/diverge_vcr1/train_seed2026080603.jsonl}
DEVELOPMENT=${DEVELOPMENT:-$BASE/artifacts/reasoning/diverge_vcr1/development_seed2026080603.jsonl}
REPORT=${REPORT:-$BASE/artifacts/reasoning/diverge_vcr1/board_seed2026080603.report.json}

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
test "$(sha256sum "$MODEL/tokenizer_config.json" | awk '{print $1}')" = \
  59a64b39f05ee0acf7479bbccf99a44d9e481c1a4a494eb03838ae3c35eb29be
test "$(sha256sum "$SOURCE" | awk '{print $1}')" = \
  38d582fa6f5626f34d7f390e243bc4f17c4114ce06d83fe0feca755bd7b88ba6
test ! -e "$TRAIN"
test ! -e "$DEVELOPMENT"
test ! -e "$REPORT"
mkdir -p "$(dirname "$TRAIN")"
trap 'for path in "$TRAIN" "$DEVELOPMENT" "$REPORT"; do test ! -e "$path" || chmod a-w "$path"; done' EXIT

echo "[vcr1-board] node=${SLURMD_NODENAME:-?} code=$CODE_ROOT commit=$SOURCE_COMMIT"
"$PYTHON" -m pytest -q "$CODE_ROOT/pipeline/test_build_diverge_vcr1_board.py" "$CODE_ROOT/train/test_diverge_vcr1_data.py"
"$PYTHON" -m py_compile \
  "$CODE_ROOT/pipeline/build_diverge_vcr1_board.py" \
  "$CODE_ROOT/train/diverge_vcr1_data.py"
"$PYTHON" "$CODE_ROOT/pipeline/build_diverge_vcr1_board.py" \
  --model-root "$MODEL" \
  --source "$SOURCE" \
  --source-sha256 38d582fa6f5626f34d7f390e243bc4f17c4114ce06d83fe0feca755bd7b88ba6 \
  --train-output "$TRAIN" \
  --development-output "$DEVELOPMENT" \
  --report "$REPORT" \
  --exclude-eval "$BASE/artifacts/evals/math500.jsonl" \
  --exclude-eval "$BASE/artifacts/product_reasoning/router_outcomes/science_heldout2000_77f4cb1.jsonl" \
  --max-sequence-length 4096 \
  --workspace-slots 8 \
  --train-per-group 1400 \
  --development-per-group 160 \
  --minimum-train-per-group 400 \
  --minimum-development-per-group 40 \
  --development-fraction 0.125 \
  --seed 2026080603

sha256sum "$TRAIN" "$DEVELOPMENT" "$REPORT"
echo "[vcr1-board] completed"
