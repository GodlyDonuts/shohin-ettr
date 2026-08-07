#!/bin/bash
#SBATCH -J confirm-gti1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:30:00
#SBATCH --exclude=evc33,evc43
#SBATCH -o /lustre/fs1/home/%u/shohin/logs/confirm_gti1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

ROOT=/lustre/fs1/home/sa305415/shohin
BASE_CODE_ROOT=${BASE_CODE_ROOT:?set immutable qualified base code root}
OVERLAY_ROOT=${OVERLAY_ROOT:?set immutable GTI1 overlay root}
ASSESSMENT=${ASSESSMENT:?set matched GTI1 assessment}
SMOL_DEV=${SMOL_DEV:?set matched SmolLM2 development result}
MODEL_ROOT=${MODEL_ROOT:?set qualified SmolLM2 GTI1 model root}
OUTPUT=${OUTPUT:?set sealed GTI1 confirmation result}
PYTHON=${PYTHON:-$ROOT/envs/product-reasoning-b3a3603-r2/bin/python}
PARENT=$ROOT/train/ettr_smollm2_control_parent_a2026072801_7881d8e/joint-model-final.pt
PARENT_SHA256=8196f810a31e0abe7f3bf0eae0a37b103195f109b7a8e962c7b74b5710c98a02
TOKENIZER=$ROOT/artifacts/external/smollm2_135m_instruct_83212e1e/tokenizer.json
TOKENIZER_SHA256=9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c
DATA=$ROOT/artifacts/reasoning/diverge_pqi1/data_defc87f_r1/confirmation_board.jsonl
DATA_SHA256=27f198680cc7bcd7e0203949fe4dee1658fc057fe75302b7e2d43d74321201b8
CHECKPOINT=$MODEL_ROOT/checkpoint.pt

export PYTHONPATH="$OVERLAY_ROOT/train:$BASE_CODE_ROOT/train:$BASE_CODE_ROOT/pipeline"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$ROOT/logs"
cd "$BASE_CODE_ROOT"

for path in "$ASSESSMENT" "$SMOL_DEV" "$CHECKPOINT" "$PARENT" "$TOKENIZER" "$DATA"; do
  test -f "$path" || { echo "missing GTI1 confirmation input: $path" >&2; exit 2; }
done
test ! -e "$OUTPUT" || { echo "refusing existing GTI1 confirmation" >&2; exit 2; }

"$PYTHON" - "$ASSESSMENT" "$SMOL_DEV" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

assessment_path = Path(sys.argv[1])
smol_path = Path(sys.argv[2])
assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
if assessment.get("schema") != "shohin-diverge-gti1-assessment-v1":
    raise SystemExit("GTI1 assessment schema differs")
if assessment.get("status") != "pass" or assessment["promotion_gate"]["passed"] is not True:
    raise SystemExit("GTI1 development gate did not pass")
if assessment["inputs"]["smollm2"]["path"] != str(smol_path):
    raise SystemExit("GTI1 qualified arm path differs")
digest = hashlib.sha256(smol_path.read_bytes()).hexdigest()
if assessment["inputs"]["smollm2"]["sha256"] != digest:
    raise SystemExit("GTI1 qualified development result hash differs")
PY

CHECKPOINT_SHA256=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
"$PYTHON" "$OVERLAY_ROOT/train/eval_diverge_gti1.py" \
  --checkpoint "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --base "$PARENT" --base-sha256 "$PARENT_SHA256" \
  --tokenizer "$TOKENIZER" --tokenizer-sha256 "$TOKENIZER_SHA256" \
  --data "$DATA" --data-sha256 "$DATA_SHA256" \
  --board-type confirmation --output "$OUTPUT" --batch-size 64 --device cuda

"$PYTHON" - "$OUTPUT" <<'PY'
import json
from pathlib import Path
import sys

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if result.get("schema") != "shohin-diverge-gti1-evaluation-v1":
    raise SystemExit("GTI1 confirmation schema differs")
if result.get("board_type") != "confirmation" or result.get("status") != "pass":
    raise SystemExit("GTI1 sealed confirmation failed")
PY

echo "[confirm-gti1] sealed confirmation passed"
