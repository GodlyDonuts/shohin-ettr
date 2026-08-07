#!/bin/bash
#SBATCH -J confirm-cgl1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH --exclude=evc33,evc43
#SBATCH -o /lustre/fs1/home/%u/shohin/logs/confirm_diverge_cgl1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
BASE_CODE_ROOT=${BASE_CODE_ROOT:?set immutable qualified base code root}
OVERLAY_ROOT=${OVERLAY_ROOT:?set immutable CGL1 confirmation overlay}
ASSESSMENT=${ASSESSMENT:?set immutable passing CGL1 assessment}
ASSESSMENT_SHA256=${ASSESSMENT_SHA256:?set CGL1 assessment SHA-256}
RESULTS_ROOT=${RESULTS_ROOT:?set immutable CGL1 development results root}
DATA=${DATA:?set fresh balanced CGL1 confirmation board}
DATA_SHA256=${DATA_SHA256:?set CGL1 confirmation board SHA-256}
OUTPUT=${OUTPUT:?set fresh CGL1 confirmation report}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

test "$(sha256sum "$ASSESSMENT" | awk '{print $1}')" = "$ASSESSMENT_SHA256"
test "$(sha256sum "$DATA" | awk '{print $1}')" = "$DATA_SHA256"
read -r SELECTED < <("$PYTHON" - "$ASSESSMENT" <<'PY'
import json
from pathlib import Path
import sys
report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("status") != "pass" or report.get("selected_treatment") not in {
    "shohin", "smollm2"
}:
    raise SystemExit("CGL1 assessment did not admit confirmation")
print(report["selected_treatment"])
PY
)

case "$SELECTED" in
  shohin)
    ARM=$RESULTS_ROOT/shohin
    PARENT=$BASE/train/flagship_out/ckpt_0300000.pt
    PARENT_SHA256=211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6
    TOKENIZER=$BASE/artifacts/shohin-tok-32k.json
    TOKENIZER_SHA256=87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4
    ;;
  smollm2)
    ARM=$RESULTS_ROOT/smollm2
    PARENT=$BASE/train/ettr_smollm2_control_parent_a2026072801_7881d8e/joint-model-final.pt
    PARENT_SHA256=8196f810a31e0abe7f3bf0eae0a37b103195f109b7a8e962c7b74b5710c98a02
    TOKENIZER=$BASE/artifacts/external/smollm2_135m_instruct_83212e1e/tokenizer.json
    TOKENIZER_SHA256=9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c
    ;;
esac
CHECKPOINT=$ARM/model/checkpoint.pt
CHECKPOINT_SHA256=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
test ! -e "$OUTPUT" || { echo "refusing existing CGL1 confirmation" >&2; exit 2; }

export PYTHONPATH="$OVERLAY_ROOT/train:$BASE_CODE_ROOT/train:$BASE_CODE_ROOT/pipeline"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs"
cd "$BASE_CODE_ROOT"

echo "[confirm-cgl1] node=${SLURMD_NODENAME:-?} selected=$SELECTED"
nvidia-smi -L
"$PYTHON" "$OVERLAY_ROOT/train/eval_diverge_cgl1.py" \
  --checkpoint "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --base "$PARENT" --base-sha256 "$PARENT_SHA256" \
  --tokenizer "$TOKENIZER" --tokenizer-sha256 "$TOKENIZER_SHA256" \
  --data "$DATA" --data-sha256 "$DATA_SHA256" \
  --board-type confirmation --output "$OUTPUT" --batch-size 32 --device cuda

"$PYTHON" - "$OUTPUT" <<'PY'
import json
from pathlib import Path
import sys
result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if result.get("status") != "pass" or not result["promotion_gate"]["passed"]:
    raise SystemExit("CGL1 sealed confirmation failed")
PY

echo "[confirm-cgl1] sealed confirmation passed"
