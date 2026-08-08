#!/bin/bash
#SBATCH --job-name=idr1-train
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/idr1_train_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/idr1_train_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${RUNTIME_MANIFEST_SHA256:?RUNTIME_MANIFEST_SHA256 is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${MODEL_REVISION:?MODEL_REVISION is required}"
: "${MODEL_CONFIG_SHA256:?MODEL_CONFIG_SHA256 is required}"
: "${WARM_START:?WARM_START is required}"
: "${WARM_START_SHA256:?WARM_START_SHA256 is required}"
: "${DATA:?DATA is required}"
: "${DATA_SHA256:?DATA_SHA256 is required}"
: "${DATA_REPORT:?DATA_REPORT is required}"
: "${DATA_REPORT_SHA256:?DATA_REPORT_SHA256 is required}"
: "${OUTPUT:?OUTPUT is required}"

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

test -x "$PYTHON"
for path in \
  "$RUNTIME/train/hf_product_reasoning_train.py" \
  "$RUNTIME/train/integrated_reasoning_workspace.py" \
  "$MODEL_ROOT/config.json" \
  "$WARM_START" \
  "$DATA" \
  "$DATA_REPORT"; do
  test -f "$path"
done
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$WARM_START" | cut -d' ' -f1)" = "$WARM_START_SHA256"
test "$(sha256sum "$DATA" | cut -d' ' -f1)" = "$DATA_SHA256"
test "$(sha256sum "$DATA_REPORT" | cut -d' ' -f1)" = "$DATA_REPORT_SHA256"
test ! -e "$OUTPUT"

"$PYTHON" - "$DATA" "$DATA_REPORT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

data, report_path = map(Path, sys.argv[1:])
report = json.loads(report_path.read_text(encoding="utf-8"))
if report.get("schema") != "shohin-idr1-revision-data-report-v1":
    raise SystemExit("IDR1 data report schema differs")
if report.get("internal_draft_visible") is not True:
    raise SystemExit("IDR1 internal-draft boundary differs")
if report.get("external_candidate_text_visible") is not False:
    raise SystemExit("IDR1 external-candidate boundary differs")
expected = report.get("outputs", {}).get("train", {})
if Path(expected.get("path", "")).resolve() != data.resolve():
    raise SystemExit("IDR1 train path differs")
if expected.get("sha256") != hashlib.sha256(data.read_bytes()).hexdigest():
    raise SystemExit("IDR1 train hash differs")
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
"$PYTHON" "$RUNTIME/train/hf_product_reasoning_train.py" \
  --model-root "$EFFECTIVE_MODEL_ROOT" \
  --model-source-root "$SOURCE_MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --model-loader multimodal \
  --data "$DATA" \
  --output "$OUTPUT" \
  --warm-start-checkpoint "$WARM_START" \
  --arm baseline \
  --updates 256 \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --max-rows 10000 \
  --max-sequence-length 4096 \
  --learning-rate 2e-5 \
  --lora-layers 4 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --seed 2026080815 \
  --data-seed 2026080814 \
  --log-interval 8 \
  --checkpoint-interval 256

test -f "$OUTPUT/checkpoint_0000256.pt"
test -f "$OUTPUT/report.json"
chmod -R a-w "$OUTPUT"
