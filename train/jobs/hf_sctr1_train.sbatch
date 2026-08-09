#!/bin/bash
#SBATCH --job-name=sctr1-train
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/sctr1_train_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/sctr1_train_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${RUNTIME_MANIFEST_SHA256:?RUNTIME_MANIFEST_SHA256 is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${MODEL_REVISION:?MODEL_REVISION is required}"
: "${MODEL_CONFIG_SHA256:?MODEL_CONFIG_SHA256 is required}"
: "${DATA:?DATA is required}"
: "${DATA_SHA256:?DATA_SHA256 is required}"
: "${DATA_REPORT:?DATA_REPORT is required}"
: "${DATA_REPORT_SHA256:?DATA_REPORT_SHA256 is required}"
: "${OUTPUT:?OUTPUT is required}"
MODEL_LOADER=${MODEL_LOADER:-auto}
UPDATES=${UPDATES:-256}
TRAIN_KIND=${TRAIN_KIND:-selective_commit}
DATA_KEY=${DATA_KEY:-train}

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

test -x "$PYTHON"
for path in \
  "$RUNTIME/train/hf_product_reasoning_train.py" \
  "$RUNTIME/train/integrated_reasoning_workspace.py" \
  "$MODEL_ROOT/config.json" "$DATA" "$DATA_REPORT"; do
  test -f "$path"
done
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = \
  "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
printf '%s  %s\n' "$MODEL_CONFIG_SHA256" "$MODEL_ROOT/config.json" \
  | sha256sum -c - >/dev/null
printf '%s  %s\n' "$DATA_SHA256" "$DATA" | sha256sum -c - >/dev/null
printf '%s  %s\n' "$DATA_REPORT_SHA256" "$DATA_REPORT" \
  | sha256sum -c - >/dev/null
test ! -e "$OUTPUT"

"$PYTHON" - "$DATA" "$DATA_REPORT" "$TRAIN_KIND" "$DATA_KEY" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

data, report_path = map(Path, sys.argv[1:3])
train_kind = sys.argv[3]
data_key = sys.argv[4]
report = json.loads(report_path.read_text(encoding="utf-8"))
if data_key not in ("train", "train_shuffled"):
    raise SystemExit("temporal-role data key differs")
if data_key == "train_shuffled" and train_kind != "selective_commit":
    raise SystemExit("only selective commitment has shuffled supervision")
expected = report.get("outputs", {}).get(data_key, {})
schemas = {
    "selective_commit": "shohin-sctr1-selective-commit-data-report-v1",
    "always_revise": "shohin-idr1-revision-data-report-v1",
    "independent_commitment": "shohin-idr1-revision-data-report-v1",
}
if train_kind not in schemas or report.get("schema") != schemas[train_kind]:
    raise SystemExit("temporal-role data report schema differs")
if train_kind == "selective_commit" and (
    report.get("commitment") != "whole_draft_or_whole_revision"
):
    raise SystemExit("SCTR1 commitment differs")
if expected.get("rows") != 9655:
    raise SystemExit("SCTR1 train cardinality differs")
if Path(expected.get("path", "")).resolve() != data.resolve():
    raise SystemExit("SCTR1 train path differs")
if expected.get("sha256") != hashlib.sha256(data.read_bytes()).hexdigest():
    raise SystemExit("SCTR1 train hash differs")
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
MASK_ARGS=()
if [[ "$TRAIN_KIND" == "independent_commitment" ]]; then
  MASK_ARGS+=(--mask-internal-draft)
elif [[ "$TRAIN_KIND" != "selective_commit" && "$TRAIN_KIND" != "always_revise" ]]; then
  echo "TRAIN_KIND differs" >&2
  exit 4
fi
"$PYTHON" "$RUNTIME/train/hf_product_reasoning_train.py" \
  --model-root "$EFFECTIVE_MODEL_ROOT" \
  --model-source-root "$SOURCE_MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --model-loader "$MODEL_LOADER" \
  --data "$DATA" \
  --output "$OUTPUT" \
  --arm baseline \
  --updates "$UPDATES" \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --max-rows 10000 \
  --max-sequence-length 4096 \
  --learning-rate 2e-5 \
  --lora-layers 4 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --seed 2026080822 \
  --data-seed 2026080821 \
  --log-interval 8 \
  --checkpoint-interval "$UPDATES" \
  "${MASK_ARGS[@]}"

printf -v checkpoint '%s/checkpoint_%07d.pt' "$OUTPUT" "$UPDATES"
test -f "$checkpoint"
test -f "$OUTPUT/report.json"
chmod -R a-w "$OUTPUT"
