#!/bin/bash
#SBATCH --job-name=aqc1-commit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/aqc1_commit_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/aqc1_commit_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${RUNTIME_MANIFEST_SHA256:?RUNTIME_MANIFEST_SHA256 is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${MODEL_REVISION:?MODEL_REVISION is required}"
: "${MODEL_CONFIG_SHA256:?MODEL_CONFIG_SHA256 is required}"
: "${ADAPTER_CHECKPOINT:?ADAPTER_CHECKPOINT is required}"
: "${ADAPTER_CHECKPOINT_SHA256:?ADAPTER_CHECKPOINT_SHA256 is required}"
: "${PAIRS:?PAIRS is required}"
: "${PAIRS_SHA256:?PAIRS_SHA256 is required}"
: "${PAIRS_REPORT:?PAIRS_REPORT is required}"
: "${PAIRS_REPORT_SHA256:?PAIRS_REPORT_SHA256 is required}"
: "${ARM:?ARM is required}"
: "${SEED:?SEED is required}"
: "${OUTPUT:?OUTPUT is required}"
UPDATES=${UPDATES:-128}

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

test -x "$PYTHON"
for path in \
  "$RUNTIME/train/hf_aqc1_train_commit.py" \
  "$RUNTIME/train/hf_cvg1_completion_verifier.py" \
  "$RUNTIME/train/hf_product_reasoning_eval.py" \
  "$RUNTIME/train/hf_product_reasoning_train.py" \
  "$RUNTIME/train/integrated_reasoning_workspace.py" \
  "$MODEL_ROOT/config.json" \
  "$ADAPTER_CHECKPOINT" \
  "$PAIRS" \
  "$PAIRS_REPORT"; do
  test -f "$path"
done
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = \
  "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = \
  "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$ADAPTER_CHECKPOINT" | cut -d' ' -f1)" = \
  "$ADAPTER_CHECKPOINT_SHA256"
test "$(sha256sum "$PAIRS" | cut -d' ' -f1)" = "$PAIRS_SHA256"
test "$(sha256sum "$PAIRS_REPORT" | cut -d' ' -f1)" = \
  "$PAIRS_REPORT_SHA256"
test ! -e "$OUTPUT"

"$PYTHON" - "$PAIRS" "$PAIRS_REPORT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

pairs, report_path = map(Path, sys.argv[1:])
report = json.loads(report_path.read_text(encoding="utf-8"))
if report.get("schema") != "shohin-aqc1-pair-report-v1":
    raise SystemExit("AQC1 pair report schema differs")
if report.get("status") != "complete":
    raise SystemExit("AQC1 pair report is incomplete")
if Path(report.get("output", "")).resolve() != pairs.resolve():
    raise SystemExit("AQC1 pair path differs")
if report.get("output_sha256") != hashlib.sha256(pairs.read_bytes()).hexdigest():
    raise SystemExit("AQC1 pair hash differs")
PY

SOURCE_MODEL_ROOT=$MODEL_ROOT
LOCAL_PARENT=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
EFFECTIVE_MODEL_ROOT=$LOCAL_PARENT/model
mkdir -p "$EFFECTIVE_MODEL_ROOT" logs
cp -a "$SOURCE_MODEL_ROOT"/. "$EFFECTIVE_MODEL_ROOT"/

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"

nvidia-smi
"$PYTHON" "$RUNTIME/train/hf_aqc1_train_commit.py" \
  --arm "$ARM" \
  --model-root "$EFFECTIVE_MODEL_ROOT" \
  --model-source-root "$SOURCE_MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --model-loader multimodal \
  --adapter-checkpoint "$ADAPTER_CHECKPOINT" \
  --pairs "$PAIRS" \
  --output "$OUTPUT" \
  --updates "$UPDATES" \
  --gradient-accumulation 8 \
  --head-width 512 \
  --projection-width 256 \
  --max-sequence-length 3072 \
  --evaluation-batch-pairs 2 \
  --backbone-learning-rate 2e-6 \
  --head-learning-rate 2e-4 \
  --tie-loss-weight 0.25 \
  --seed "$SEED"

test -f "$OUTPUT/commit.pt"
test -f "$OUTPUT/report.json"
sha256sum "$OUTPUT/commit.pt" "$OUTPUT/report.json"
chmod -R a-w "$OUTPUT"
