#!/bin/bash
#SBATCH --job-name=q36-moe-mech
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/q36_moe_mechanics_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/q36_moe_mechanics_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION \
  MODEL_CONFIG_SHA256 BNB_ROOT BNB_MANIFEST_SHA256 DATA DATA_SHA256 OUTPUT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
for path in "$RUNTIME/train/hf_product_reasoning_train.py" \
  "$RUNTIME/train/integrated_reasoning_workspace.py" "$RUNTIME/train/ttr1_revision.py" \
  "$MODEL_ROOT/config.json" "$BNB_ROOT/SHA256SUMS" "$DATA"; do test -f "$path"; done
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$BNB_ROOT/SHA256SUMS" | cut -d' ' -f1)" = "$BNB_MANIFEST_SHA256"
(cd "$BNB_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$DATA" | cut -d' ' -f1)" = "$DATA_SHA256"
test ! -e "$OUTPUT"

export TMPDIR=$BASE/tmp/$SLURM_JOB_ID
trap 'rm -rf "$TMPDIR"' EXIT
mkdir -p "$TMPDIR" logs "$(dirname "$OUTPUT")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$BNB_ROOT:$RUNTIME/train:$RUNTIME/pipeline"
nvidia-smi
"$PYTHON" "$RUNTIME/train/hf_product_reasoning_train.py" \
  --model-root "$MODEL_ROOT" --model-source-root "$MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" --model-loader auto --quantization nf4 \
  --data "$DATA" --output "$OUTPUT" --arm ettr --updates 1 --batch-size 1 \
  --gradient-accumulation 1 --max-rows 24 --max-sequence-length 1024 \
  --learning-rate 2e-5 --lora-layers 1 --lora-rank 8 --lora-alpha 16 \
  --lora-scope token_mixer --workspace-width 256 --workspace-slots 8 \
  --recurrent-steps 2 --seed 2026080825 --data-seed 2026080824 \
  --log-interval 1 --checkpoint-interval 1
test -f "$OUTPUT/checkpoint_0000001.pt"; test -f "$OUTPUT/report.json"
"$PYTHON" - "$OUTPUT/report.json" <<'PY'
import json, sys
from pathlib import Path
report=json.loads(Path(sys.argv[1]).read_text())
if report.get("quantization") != "nf4" or report.get("lora_scope") != "token_mixer":
    raise SystemExit("Qwen3.6 mechanics metadata differs")
if report.get("updates") != 1 or report.get("peak_gpu_memory_bytes", 0) <= 0:
    raise SystemExit("Qwen3.6 mechanics did not execute")
print(json.dumps({k:report[k] for k in ("trainable_parameters","peak_gpu_memory_bytes","charged_tokens_per_second","trainable_parameter_name_sha256")},sort_keys=True))
PY
chmod -R a-w "$OUTPUT"
