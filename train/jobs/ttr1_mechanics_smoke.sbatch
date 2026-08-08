#!/bin/bash
#SBATCH --job-name=ttr1-smoke
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/ttr1_smoke_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/ttr1_smoke_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${RUNTIME_MANIFEST_SHA256:?RUNTIME_MANIFEST_SHA256 is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${MODEL_REVISION:?MODEL_REVISION is required}"
: "${MODEL_CONFIG_SHA256:?MODEL_CONFIG_SHA256 is required}"
: "${ADAPTER_CHECKPOINT:?ADAPTER_CHECKPOINT is required}"
: "${NORMALIZATION_REPORT:?NORMALIZATION_REPORT is required}"
: "${MATH_BANK:?MATH_BANK is required}"
: "${MATH_BANK_SHA256:?MATH_BANK_SHA256 is required}"
: "${SCIENCE_BANK:?SCIENCE_BANK is required}"
: "${SCIENCE_BANK_SHA256:?SCIENCE_BANK_SHA256 is required}"
: "${CODE_BANK:?CODE_BANK is required}"
: "${CODE_BANK_SHA256:?CODE_BANK_SHA256 is required}"
: "${REPORT:?REPORT is required}"

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test -x "$PYTHON"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test ! -e "$REPORT"
ADAPTER_CHECKPOINT_SHA256=$("$PYTHON" - "$ADAPTER_CHECKPOINT" "$NORMALIZATION_REPORT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

checkpoint, report_path = map(Path, sys.argv[1:])
report = json.loads(report_path.read_text(encoding="ascii"))
expected = report.get("output", {})
actual = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
if Path(expected.get("path", "")).resolve() != checkpoint.resolve():
    raise SystemExit("normalized checkpoint path differs")
if expected.get("sha256") != actual:
    raise SystemExit("normalized checkpoint SHA-256 differs")
if report.get("all_non_metadata_state_bitwise_equal") is not True:
    raise SystemExit("normalization state proof differs")
print(actual)
PY
)

SOURCE_MODEL_ROOT=$MODEL_ROOT
LOCAL_PARENT=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
EFFECTIVE_MODEL_ROOT=$LOCAL_PARENT/model
mkdir -p "$EFFECTIVE_MODEL_ROOT" "$(dirname "$REPORT")"
cp -a "$SOURCE_MODEL_ROOT"/. "$EFFECTIVE_MODEL_ROOT"/
test -f "$EFFECTIVE_MODEL_ROOT/config.json"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
nvidia-smi
"$PYTHON" "$RUNTIME/train/ttr1_mechanics_smoke.py" \
  --model-root "$EFFECTIVE_MODEL_ROOT" \
  --model-source-root "$SOURCE_MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --model-config-sha256 "$MODEL_CONFIG_SHA256" \
  --adapter-checkpoint "$ADAPTER_CHECKPOINT" \
  --adapter-checkpoint-sha256 "$ADAPTER_CHECKPOINT_SHA256" \
  --math-bank "$MATH_BANK" \
  --math-bank-sha256 "$MATH_BANK_SHA256" \
  --science-bank "$SCIENCE_BANK" \
  --science-bank-sha256 "$SCIENCE_BANK_SHA256" \
  --code-bank "$CODE_BANK" \
  --code-bank-sha256 "$CODE_BANK_SHA256" \
  --report "$REPORT"
chmod a-w "$REPORT"
sha256sum "$REPORT"
