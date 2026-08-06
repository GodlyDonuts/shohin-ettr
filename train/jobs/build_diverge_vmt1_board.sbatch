#!/bin/bash
#SBATCH -J vmt1-board
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/diverge_vmt1_board_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
SOURCE_COMMIT=${SOURCE_COMMIT:?set immutable SOURCE_COMMIT}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL=$BASE/artifacts/external/qwen3.5-0.8b-2fc0636
SOURCE=${SOURCE:-$BASE/artifacts/product_reasoning/candidate_rescore_6e5016c_r1/fresh8192.candidates.jsonl}
OUTPUT=${OUTPUT:-$BASE/artifacts/reasoning/diverge_vmt1/fit_board_seed2026080602.jsonl}
REPORT=${REPORT:-$BASE/artifacts/reasoning/diverge_vmt1/fit_board_seed2026080602.report.json}

export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" logs
cd "$BASE"

test "$(sha256sum "$MODEL/config.json" | awk '{print $1}')" = \
  b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204
test "$(sha256sum "$MODEL/tokenizer.json" | awk '{print $1}')" = \
  5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42
test "$(sha256sum "$MODEL/tokenizer_config.json" | awk '{print $1}')" = \
  49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c
test "$(sha256sum "$SOURCE" | awk '{print $1}')" = \
  20a496867c1afc46d094a1ee2762cc553bd0460bd2915cc7e60d9c53025aa816
test ! -e "$OUTPUT" || { echo "refusing existing board: $OUTPUT" >&2; exit 2; }
test ! -e "$REPORT" || { echo "refusing existing report: $REPORT" >&2; exit 2; }
mkdir -p "$(dirname "$OUTPUT")"
trap 'test ! -e "$OUTPUT" || chmod a-w "$OUTPUT"; test ! -e "$REPORT" || chmod a-w "$REPORT"' EXIT

echo "[vmt1-board] node=${SLURMD_NODENAME:-?} code=$CODE_ROOT commit=$SOURCE_COMMIT"
"$PYTHON" "$CODE_ROOT/pipeline/test_build_diverge_vmt1_board.py"
"$PYTHON" -m py_compile "$CODE_ROOT/pipeline/build_diverge_vmt1_board.py"
"$PYTHON" "$CODE_ROOT/pipeline/build_diverge_vmt1_board.py" \
  --source "$SOURCE" \
  --output "$OUTPUT" \
  --report "$REPORT" \
  --model-root "$MODEL" \
  --per-cell 4 \
  --max-sequence-length 1024 \
  --workspace-slots 8 \
  --seed 2026080602

sha256sum "$OUTPUT" "$REPORT"
echo "[vmt1-board] completed"
