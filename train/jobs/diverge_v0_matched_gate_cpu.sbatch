#!/bin/bash
#SBATCH -J diverge-a-g-cpu
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/diverge_a_g_cpu_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
SOURCE_COMMIT=${SOURCE_COMMIT:?set immutable SOURCE_COMMIT}
CHECKPOINT=${CHECKPOINT:?set immutable role-copy checkpoint}
SOURCE_REPORT=${SOURCE_REPORT:?set immutable role-copy report}
OUT=${OUT:?set isolated matched-gate report}
SEED=${SEED:?set role-copy seed}
PYTHON=$BASE/miniforge3/bin/python3
PARENT=$BASE/train/ettr_smollm2_control_parent_a2026072801_7881d8e/joint-model-final.pt
TOKENIZER=$BASE/artifacts/external/smollm2_135m_instruct_83212e1e/tokenizer.json
BOARD_SEED=${BOARD_SEED:-202608056000}
REPETITIONS=${REPETITIONS:-4}

export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$BASE"

test "$(sha256sum "$PARENT" | awk '{print $1}')" = \
  8196f810a31e0abe7f3bf0eae0a37b103195f109b7a8e962c7b74b5710c98a02
test "$(sha256sum "$TOKENIZER" | awk '{print $1}')" = \
  9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c
test -f "$CHECKPOINT"
test -f "$SOURCE_REPORT"
test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
mkdir -p "$(dirname "$OUT")"

echo "[diverge-a-g-cpu] node=${SLURMD_NODENAME:-?} code=$CODE_ROOT commit=$SOURCE_COMMIT seed=$SEED"
"$PYTHON" -m pytest -q \
  "$CODE_ROOT/train/test_diverge_v0.py" \
  "$CODE_ROOT/train/test_diverge_v0_matched_gate.py"
"$PYTHON" "$CODE_ROOT/train/diverge_v0_matched_gate.py" \
  --base "$PARENT" \
  --tokenizer "$TOKENIZER" \
  --checkpoint "$CHECKPOINT" \
  --source-report "$SOURCE_REPORT" \
  --output "$OUT" \
  --seed "$SEED" \
  --board-seed "$BOARD_SEED" \
  --repetitions "$REPETITIONS" \
  --layer 17 \
  --width 192 \
  --char-width 48 \
  --profile-batch-size 4 \
  --threads 4 \
  --device cpu

chmod a-w "$OUT"
echo "[diverge-a-g-cpu] completed"
