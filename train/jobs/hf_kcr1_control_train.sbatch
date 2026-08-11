#!/bin/bash
#SBATCH --job-name=kcr1-control-fit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=06:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/kcr1_control_fit_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/kcr1_control_fit_%j.err

set -euo pipefail

for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION \
  MODEL_CONFIG_SHA256 WARM_START WARM_START_SHA256 DATA DATA_SHA256 \
  DATA_REPORT DATA_REPORT_SHA256 OUTPUT CONTROL UPDATES; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
case "$CONTROL" in
  action_permuted|constant_restart) MASK_ARGS=() ;;
  draft_hidden) MASK_ARGS=(--mask-internal-draft) ;;
  *) echo "unsupported KCR1 control" >&2; exit 4 ;;
esac
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-$UPDATES}

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
"$PYTHON" - "$DATA" "$DATA_REPORT" "$CONTROL" <<'PY'
import hashlib,json,sys
from pathlib import Path
data,report_path,control=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
report=json.loads(report_path.read_text())
if report.get("status")!="complete" or report.get("zero_truncation") is not True or report.get("holdout_used") is not False: raise SystemExit("KCR1 control report differs")
if control=="draft_hidden":
    expected=report.get("output",{})
    if report.get("schema")!="shohin-kcr1-branch-data-report-v1": raise SystemExit("KCR1 hidden source differs")
else:
    expected=report.get("outputs",{}).get(control,{})
    if report.get("schema")!="shohin-kcr1-control-data-report-v1": raise SystemExit("KCR1 control source differs")
if expected.get("rows")!=33654 or Path(expected.get("path","")).resolve()!=data.resolve(): raise SystemExit("KCR1 control path differs")
if expected.get("sha256")!=hashlib.sha256(data.read_bytes()).hexdigest(): raise SystemExit("KCR1 control hash differs")
PY

LOCAL_PARENT=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
EFFECTIVE_MODEL_ROOT=$LOCAL_PARENT/model
mkdir -p "$EFFECTIVE_MODEL_ROOT" logs
cp -a "$MODEL_ROOT"/. "$EFFECTIVE_MODEL_ROOT"/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
nvidia-smi
"$PYTHON" "$RUNTIME/train/hf_product_reasoning_train.py" \
  --model-root "$EFFECTIVE_MODEL_ROOT" --model-source-root "$MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" --model-loader multimodal \
  --data "$DATA" --output "$OUTPUT" --warm-start-checkpoint "$WARM_START" \
  --arm baseline --loss-mode kcr1_action_payload "${MASK_ARGS[@]}" \
  --updates "$UPDATES" --batch-size 1 --gradient-accumulation 8 \
  --max-rows 100000 --max-sequence-length 4096 --learning-rate 2e-5 \
  --lora-layers 4 --lora-rank 8 --lora-alpha 16 --seed 2026080921 \
  --data-seed 2026080920 --log-interval 8 \
  --checkpoint-interval "$CHECKPOINT_INTERVAL"
test -f "$OUTPUT/checkpoint_$(printf '%07d' "$UPDATES").pt"
test -f "$OUTPUT/report.json"
chmod -R a-w "$OUTPUT"
