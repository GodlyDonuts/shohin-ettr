#!/bin/bash
#SBATCH --job-name=vte1-train
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/vte1_train_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/vte1_train_%j.err

set -euo pipefail

for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION \
  MODEL_CONFIG_SHA256 WARM_START WARM_START_SHA256 DATA DATA_SHA256 \
  DATA_REPORT DATA_REPORT_SHA256 OUTPUT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL_LOADER=${MODEL_LOADER:-multimodal}
UPDATES=${UPDATES:-256}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-256}
MASK_INTERNAL_DRAFT=${MASK_INTERNAL_DRAFT:-0}

test -x "$PYTHON"
for required_path in "$RUNTIME/train/hf_product_reasoning_train.py" \
  "$RUNTIME/train/integrated_reasoning_workspace.py" "$MODEL_ROOT/config.json" \
  "$WARM_START" "$DATA" "$DATA_REPORT"; do test -f "$required_path"; done
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = \
  "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = \
  "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$WARM_START" | cut -d' ' -f1)" = "$WARM_START_SHA256"
test "$(sha256sum "$DATA" | cut -d' ' -f1)" = "$DATA_SHA256"
test "$(sha256sum "$DATA_REPORT" | cut -d' ' -f1)" = "$DATA_REPORT_SHA256"
test ! -e "$OUTPUT"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
"$PYTHON" - "$DATA" "$DATA_REPORT" <<'PY'
import hashlib,json,sys
from pathlib import Path
data,report_path=Path(sys.argv[1]),Path(sys.argv[2])
report=json.loads(report_path.read_text())
output=report.get("output",{})
if report.get("schema")!="shohin-vte1-equivalence-data-report-v1" or report.get("status")!="complete": raise SystemExit("VTE1 report differs")
if report.get("source_rows")!=11220 or report.get("admitted_sources",0)<10098: raise SystemExit("VTE1 source admission differs")
if report.get("presentations")!=3*report.get("admitted_sources",-1): raise SystemExit("VTE1 presentation balance differs")
if report.get("equivalence_candidates",0)<report.get("presentations",0): raise SystemExit("VTE1 equivalence sets differ")
if report.get("zero_truncation") is not True or report.get("max_sequence_length")!=4096: raise SystemExit("VTE1 token custody differs")
if report.get("holdout_used") is not False or report.get("runtime_fields")!="question responses".split(): raise SystemExit("VTE1 runtime fields differ")
if report.get("assessor_fields_visible_to_model") is not False: raise SystemExit("VTE1 assessor field leaked")
if Path(output.get("path","")).resolve()!=data.resolve(): raise SystemExit("VTE1 train path differs")
if output.get("sha256")!=hashlib.sha256(data.read_bytes()).hexdigest(): raise SystemExit("VTE1 train hash differs")
PY

LOCAL_PARENT=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
EFFECTIVE_MODEL_ROOT=$LOCAL_PARENT/model
mkdir -p "$EFFECTIVE_MODEL_ROOT" logs
cp -a "$MODEL_ROOT"/. "$EFFECTIVE_MODEL_ROOT"/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
nvidia-smi
mask_args=()
if test "$MASK_INTERNAL_DRAFT" = 1; then mask_args+=(--mask-internal-draft); fi
"$PYTHON" "$RUNTIME/train/hf_product_reasoning_train.py" \
  --model-root "$EFFECTIVE_MODEL_ROOT" --model-source-root "$MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" --model-loader "$MODEL_LOADER" \
  --data "$DATA" --output "$OUTPUT" --warm-start-checkpoint "$WARM_START" \
  --arm baseline --loss-mode vte1_equivalence "${mask_args[@]}" \
  --updates "$UPDATES" --batch-size 1 --gradient-accumulation 8 \
  --max-rows 100000 --max-sequence-length 4096 --learning-rate 2e-5 \
  --lora-layers 4 --lora-rank 8 --lora-alpha 16 --seed 2026081021 \
  --data-seed 2026081020 --log-interval 8 \
  --checkpoint-interval "$CHECKPOINT_INTERVAL"
checkpoint="$OUTPUT/checkpoint_$(printf '%07d' "$UPDATES").pt"
test -f "$checkpoint"
test -f "$OUTPUT/report.json"
"$PYTHON" - "$OUTPUT/report.json" "$UPDATES" "$MASK_INTERNAL_DRAFT" <<'PY'
import json,sys
report=json.load(open(sys.argv[1]))
if report.get("status")!="complete" or report.get("updates")!=int(sys.argv[2]): raise SystemExit("VTE1 fit incomplete")
if report.get("loss_mode")!="vte1_equivalence" or report.get("vte1_temperature")!=0.1: raise SystemExit("VTE1 objective differs")
if report.get("trainable_parameters")!=2704896: raise SystemExit("VTE1 parameter geometry differs")
if report.get("mask_internal_draft")!=(sys.argv[3]=="1"): raise SystemExit("VTE1 draft intervention differs")
PY
sha256sum "$checkpoint" "$OUTPUT/report.json"
chmod -R a-w "$OUTPUT"
