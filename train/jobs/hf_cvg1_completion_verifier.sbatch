#!/bin/bash
#SBATCH --job-name=cvg1-critic
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/cvg1_critic_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/cvg1_critic_%j.err

set -euo pipefail

: "${RUNTIME:?RUNTIME is required}"
: "${RUNTIME_MANIFEST_SHA256:?RUNTIME_MANIFEST_SHA256 is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${MODEL_REVISION:?MODEL_REVISION is required}"
: "${MODEL_CONFIG_SHA256:?MODEL_CONFIG_SHA256 is required}"
: "${ADAPTER_CHECKPOINT:?ADAPTER_CHECKPOINT is required}"
: "${ADAPTER_CHECKPOINT_SHA256:?ADAPTER_CHECKPOINT_SHA256 is required}"
: "${PAIRS:?PAIRS is required}"
: "${PAIRS_REPORT:?PAIRS_REPORT is required}"
: "${OUTPUT:?OUTPUT is required}"

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

test -x "$PYTHON"
for path in \
  "$RUNTIME/train/hf_cvg1_completion_verifier.py" \
  "$RUNTIME/train/hf_product_reasoning_eval.py" \
  "$RUNTIME/train/hf_product_reasoning_train.py" \
  "$RUNTIME/train/integrated_reasoning_workspace.py" \
  "$RUNTIME/train/jobs/hf_cvg1_completion_verifier.sbatch" \
  "$MODEL_ROOT/config.json" \
  "$ADAPTER_CHECKPOINT" \
  "$PAIRS" \
  "$PAIRS_REPORT"; do
  test -f "$path"
done
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$ADAPTER_CHECKPOINT" | cut -d' ' -f1)" = "$ADAPTER_CHECKPOINT_SHA256"
test ! -e "$OUTPUT"

"$PYTHON" - "$PAIRS" "$PAIRS_REPORT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

pairs, report_path = map(Path, sys.argv[1:])
report = json.loads(report_path.read_text(encoding="utf-8"))
if report.get("schema") != "shohin-cvg1-whole-lineage-pairs-v1":
    raise SystemExit("CVG1 merge report schema differs")
if report.get("status") != "complete":
    raise SystemExit("CVG1 merge report is incomplete")
if Path(report.get("output", "")).resolve() != pairs.resolve():
    raise SystemExit("CVG1 pair path differs from merge report")
digest = hashlib.sha256(pairs.read_bytes()).hexdigest()
if report.get("output_sha256") != digest:
    raise SystemExit("CVG1 pair hash differs from merge report")
for outcome in ("base_only", "both_correct", "both_wrong", "expert_only"):
    if int(report.get("outcome_counts", {}).get(outcome, 0)) <= 0:
        raise SystemExit(f"CVG1 corpus lacks outcome class: {outcome}")
PY

SOURCE_MODEL_ROOT=$MODEL_ROOT
LOCAL_PARENT=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
EFFECTIVE_MODEL_ROOT=$LOCAL_PARENT/model
mkdir -p "$EFFECTIVE_MODEL_ROOT" logs
cp -a "$SOURCE_MODEL_ROOT"/. "$EFFECTIVE_MODEL_ROOT"/
test -f "$EFFECTIVE_MODEL_ROOT/config.json"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"

nvidia-smi
"$PYTHON" "$RUNTIME/train/hf_cvg1_completion_verifier.py" \
  --model-root "$EFFECTIVE_MODEL_ROOT" \
  --model-source-root "$SOURCE_MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --adapter-checkpoint "$ADAPTER_CHECKPOINT" \
  --pairs "$PAIRS" \
  --output "$OUTPUT" \
  --updates 256 \
  --gradient-accumulation 8 \
  --head-width 512 \
  --max-sequence-length 2048 \
  --backbone-learning-rate 2e-6 \
  --head-learning-rate 2e-4 \
  --seed 2026080716

test -f "$OUTPUT/verifier.pt"
test -f "$OUTPUT/report.json"
"$PYTHON" - "$OUTPUT/report.json" <<'PY'
import json
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("schema") != "shohin-cvg1-completion-verifier-report-v1":
    raise SystemExit("CVG1 critic report schema differs")
if report.get("status") != "complete":
    raise SystemExit("CVG1 critic report is incomplete")
if report.get("protected_adapter_unchanged") is not True:
    raise SystemExit("CVG1 critic altered its protected generator")
PY
chmod -R a-w "$OUTPUT"
