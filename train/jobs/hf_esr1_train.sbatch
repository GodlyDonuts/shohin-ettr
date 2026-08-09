#!/bin/bash
#SBATCH --job-name=esr1-train
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/esr1_train_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/esr1_train_%j.err

set -euo pipefail

for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION \
  MODEL_CONFIG_SHA256 DATA DATA_SHA256 DATA_REPORT DATA_REPORT_SHA256 OUTPUT ARM; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
case "$ARM" in syndrome|ettr) ;; *) echo "ESR1 arm differs" >&2; exit 3 ;; esac

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test -x "$PYTHON"
for path in "$RUNTIME/train/hf_product_reasoning_train.py" \
  "$RUNTIME/train/integrated_reasoning_workspace.py" "$RUNTIME/train/ttr1_revision.py" \
  "$MODEL_ROOT/config.json" "$DATA" "$DATA_REPORT"; do test -f "$path"; done
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = \
  "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
printf '%s  %s\n' "$MODEL_CONFIG_SHA256" "$MODEL_ROOT/config.json" | sha256sum -c - >/dev/null
printf '%s  %s\n' "$DATA_SHA256" "$DATA" | sha256sum -c - >/dev/null
printf '%s  %s\n' "$DATA_REPORT_SHA256" "$DATA_REPORT" | sha256sum -c - >/dev/null
test ! -e "$OUTPUT"

"$PYTHON" - "$DATA" "$DATA_REPORT" <<'PY'
import hashlib, json, sys
from pathlib import Path
data, report_path = map(Path, sys.argv[1:])
report = json.loads(report_path.read_text(encoding="utf-8"))
expected = report.get("outputs", {}).get("train", {})
if report.get("schema") != "shohin-idr1-revision-data-report-v1":
    raise SystemExit("ESR1 data report schema differs")
if expected.get("rows") != 9655 or Path(expected.get("path", "")).resolve() != data.resolve():
    raise SystemExit("ESR1 train population differs")
if expected.get("sha256") != hashlib.sha256(data.read_bytes()).hexdigest():
    raise SystemExit("ESR1 train hash differs")
PY

export TMPDIR=$BASE/tmp/$SLURM_JOB_ID
trap 'rm -rf "$TMPDIR"' EXIT
mkdir -p "$TMPDIR" logs "$(dirname "$OUTPUT")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
nvidia-smi

"$PYTHON" "$RUNTIME/train/hf_product_reasoning_train.py" \
  --model-root "$MODEL_ROOT" --model-source-root "$MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" --model-loader auto \
  --data "$DATA" --output "$OUTPUT" --arm "$ARM" \
  --updates 256 --batch-size 1 --gradient-accumulation 8 --max-rows 10000 \
  --max-sequence-length 4096 --learning-rate 2e-5 --lora-layers 4 \
  --lora-rank 8 --lora-alpha 16 --workspace-width 512 --workspace-slots 16 \
  --recurrent-steps 8 --seed 2026080822 --data-seed 2026080821 \
  --log-interval 8 --checkpoint-interval 256

test -f "$OUTPUT/checkpoint_0000256.pt"
test -f "$OUTPUT/report.json"
chmod -R a-w "$OUTPUT"
