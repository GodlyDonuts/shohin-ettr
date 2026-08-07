#!/bin/bash
#SBATCH -J confirm-qte1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH --exclude=evc33,evc43
#SBATCH -o /lustre/fs1/home/%u/shohin/logs/confirm_qte1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

ROOT=/lustre/fs1/home/sa305415/shohin
BASE_CODE_ROOT=${BASE_CODE_ROOT:?set immutable qualified base code root}
OVERLAY_ROOT=${OVERLAY_ROOT:?set immutable QTE1 overlay}
DEVELOPMENT=${DEVELOPMENT:?set frozen QTE1 development result}
OUTPUT=${OUTPUT:?set sealed QTE1 result}
PYTHON=${PYTHON:-$ROOT/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL=$ROOT/artifacts/external/qwen3.5-0.8b-2fc0636
DATA=$ROOT/artifacts/reasoning/diverge_pqi1/data_defc87f_r1/confirmation_board.jsonl
DATA_SHA256=27f198680cc7bcd7e0203949fe4dee1658fc057fe75302b7e2d43d74321201b8

export PYTHONPATH="$OVERLAY_ROOT/train:$BASE_CODE_ROOT/train:$BASE_CODE_ROOT/pipeline"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$ROOT/logs"
cd "$BASE_CODE_ROOT"

for path in "$DEVELOPMENT" "$DATA"; do
  test -f "$path" || { echo "missing QTE1 input: $path" >&2; exit 2; }
done
test ! -e "$OUTPUT" || { echo "refusing existing QTE1 confirmation" >&2; exit 2; }
test "$(sha256sum "$DEVELOPMENT" | awk '{print $1}')" = \
  cbd3dbdd26bc894d5f8ff337ecc2b4e6996d09b390a300abfe245e35981bc269
test "$(sha256sum "$MODEL/config.json" | awk '{print $1}')" = \
  b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204
test "$(sha256sum "$MODEL/model.safetensors-00001-of-00001.safetensors" | awk '{print $1}')" = \
  04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
test "$(sha256sum "$MODEL/tokenizer.json" | awk '{print $1}')" = \
  5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42

echo "[confirm-qte1] node=${SLURMD_NODENAME:-?}"
nvidia-smi -L
"$PYTHON" "$OVERLAY_ROOT/train/diagnose_diverge_gti1_qwen_entailment.py" \
  --model-root "$MODEL" \
  --data "$DATA" \
  --data-sha256 "$DATA_SHA256" \
  --board-type confirmation \
  --output "$OUTPUT" \
  --batch-size 32

"$PYTHON" - "$OUTPUT" <<'PY'
import json
from pathlib import Path
import sys

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
normal = result["controls"]["normal"]
scrub = result["controls"]["scrub_context"]
swap = result["controls"]["swap_mentions"]
exact = int(normal["overall"]["exact"])
conditions = {
    "query_at_least_765": exact >= 765,
    "every_mode_at_least_254": all(
        int(value["exact"]) >= 254 for value in normal["by_mode"].values()
    ),
    "every_renderer_at_least_127": all(
        int(value["exact"]) >= 127 for value in normal["by_renderer"].values()
    ),
    "scrub_at_most_430": int(scrub["overall"]["exact"]) <= 430,
    "scrub_loses_at_least_250": exact - int(scrub["overall"]["exact"]) >= 250,
    "swap_at_least_765": int(swap["overall"]["exact"]) >= 765,
}
print(json.dumps({"qte1_confirmation_conditions": conditions}, sort_keys=True))
if not all(conditions.values()):
    raise SystemExit("QTE1 sealed confirmation failed")
PY

echo "[confirm-qte1] sealed confirmation passed"
