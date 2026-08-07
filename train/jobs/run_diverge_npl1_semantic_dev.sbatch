#!/bin/bash
#SBATCH -J npl1-sem-dev
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/run_diverge_npl1_semantic_dev_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable NPL1 code root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set immutable source commit}
CHECKPOINT=${CHECKPOINT:?set immutable RRG1 checkpoint}
CHECKPOINT_SHA256=${CHECKPOINT_SHA256:?set RRG1 checkpoint hash}
PUBLIC_DATA=${PUBLIC_DATA:?set immutable NPL1 development public data}
PUBLIC_DATA_SHA256=${PUBLIC_DATA_SHA256:?set NPL1 public hash}
ASSESSOR_DATA=${ASSESSOR_DATA:?set immutable NPL1 development assessor data}
ASSESSOR_DATA_SHA256=${ASSESSOR_DATA_SHA256:?set NPL1 assessor hash}
OUTPUT=${OUTPUT:?set isolated NPL1 semantic result}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs"

check_hash() {
  local file=$1
  local expected=$2
  test "$(sha256sum "$file" | awk '{print $1}')" = "$expected"
}

check_hash "$CHECKPOINT" "$CHECKPOINT_SHA256"
check_hash "$PUBLIC_DATA" "$PUBLIC_DATA_SHA256"
check_hash "$ASSESSOR_DATA" "$ASSESSOR_DATA_SHA256"
test ! -e "$OUTPUT" || { echo "refusing existing NPL1 semantic output" >&2; exit 2; }

echo "[run-diverge-npl1-semantic] node=${SLURMD_NODENAME:-?} commit=$SOURCE_COMMIT"
nvidia-smi -L
"$PYTHON" -m py_compile \
  "$CODE_ROOT/train/diverge_npl1_data.py" \
  "$CODE_ROOT/train/eval_diverge_npl1_semantics.py"
"$PYTHON" "$CODE_ROOT/train/test_diverge_npl1_data.py"
"$PYTHON" "$CODE_ROOT/pipeline/test_build_diverge_npl1_development.py"
"$PYTHON" "$CODE_ROOT/train/eval_diverge_npl1_semantics.py" \
  --source-commit "$SOURCE_COMMIT" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --public-data "$PUBLIC_DATA" \
  --public-data-sha256 "$PUBLIC_DATA_SHA256" \
  --assessor-data "$ASSESSOR_DATA" \
  --assessor-data-sha256 "$ASSESSOR_DATA_SHA256" \
  --output "$OUTPUT" \
  --batch-size 512 \
  --device cuda

STATUS=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$OUTPUT")
echo "[run-diverge-npl1-semantic] completed status=$STATUS"
test "$STATUS" = pass
