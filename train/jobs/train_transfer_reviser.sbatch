#!/bin/bash
#SBATCH --job-name=transfer-reviser
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/transfer_reviser_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/transfer_reviser_%j.err

set -euo pipefail

for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION \
  MODEL_CONFIG_SHA256 MODEL_MANIFEST_SHA256 DATA DATA_SHA256 DATA_REPORT \
  DATA_REPORT_SHA256 OUTPUT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
MASK_INTERNAL_DRAFT=${MASK_INTERNAL_DRAFT:-false}
MODEL_LOADER=${MODEL_LOADER:-causal}
UPDATES=${UPDATES:-256}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-256}

BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test -x "$PYTHON"
for path in "$RUNTIME/train/hf_product_reasoning_train.py" \
  "$RUNTIME/train/integrated_reasoning_workspace.py" "$MODEL_ROOT/config.json" \
  "$MODEL_ROOT/SHA256SUMS" "$DATA" "$DATA_REPORT"; do test -f "$path"; done
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$MODEL_ROOT/SHA256SUMS" | cut -d' ' -f1)" = "$MODEL_MANIFEST_SHA256"
(cd "$MODEL_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$DATA" | cut -d' ' -f1)" = "$DATA_SHA256"
test "$(sha256sum "$DATA_REPORT" | cut -d' ' -f1)" = "$DATA_REPORT_SHA256"
test ! -e "$OUTPUT"

"$PYTHON" - "$DATA" "$DATA_REPORT" <<'PY'
import hashlib, json, sys
from pathlib import Path
data, report_path = map(Path, sys.argv[1:])
report = json.loads(report_path.read_text())
expected = report.get("outputs", {}).get("train", {})
if report.get("schema") != "shohin-idr1-revision-data-report-v1":
    raise SystemExit("revision data schema differs")
if expected.get("rows") != 9655 or expected.get("sha256") != hashlib.sha256(data.read_bytes()).hexdigest():
    raise SystemExit("revision train receipt differs")
PY

SOURCE_MODEL_ROOT=$MODEL_ROOT
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  EFFECTIVE_MODEL_ROOT=$SLURM_TMPDIR/model
  mkdir -p "$EFFECTIVE_MODEL_ROOT"
  cp -a "$SOURCE_MODEL_ROOT"/. "$EFFECTIVE_MODEL_ROOT"/
else
  EFFECTIVE_MODEL_ROOT=$SOURCE_MODEL_ROOT
fi
mkdir -p logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
nvidia-smi
MASK_ARGS=()
if [[ "$MASK_INTERNAL_DRAFT" == true ]]; then
  MASK_ARGS+=(--mask-internal-draft)
elif [[ "$MASK_INTERNAL_DRAFT" != false ]]; then
  echo "MASK_INTERNAL_DRAFT must be true or false" >&2; exit 3
fi
"$PYTHON" "$RUNTIME/train/hf_product_reasoning_train.py" \
  --model-root "$EFFECTIVE_MODEL_ROOT" --model-source-root "$SOURCE_MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" --model-loader "$MODEL_LOADER" --data "$DATA" \
  --output "$OUTPUT" --arm baseline --updates "$UPDATES" --batch-size 1 \
  --gradient-accumulation 8 --max-rows 10000 --max-sequence-length 4096 \
  --learning-rate 2e-5 --lora-layers 4 --lora-rank 8 --lora-alpha 16 \
  --lora-scope token_mixer --seed 2026080815 --data-seed 2026080814 \
  --log-interval 8 --checkpoint-interval "$CHECKPOINT_INTERVAL" "${MASK_ARGS[@]}"
CHECKPOINT="$OUTPUT/checkpoint_$(printf '%07d' "$UPDATES").pt"
test -f "$CHECKPOINT"; test -f "$OUTPUT/report.json"
"$PYTHON" - "$OUTPUT/report.json" "$CHECKPOINT" <<'PY'
import json, sys, torch
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
names=str(r.get("trainable_parameter_name_sha256", ""))
if r.get("lora_scope") != "token_mixer" or len(names) != 64:
    raise SystemExit("shared-token-mixer trainable contract differs")
payload=torch.load(sys.argv[2], map_location="cpu", weights_only=False)
trainable=sorted(payload.get("trainable_state", {}))
if not trainable or any(
    token in name.casefold()
    for name in trainable
    for token in ("experts", ".mlp.", ".gate.", "router")
):
    raise SystemExit("router or expert parameter became trainable")
print(json.dumps({"trainable_parameter_names": len(trainable), "name_sha256": names}, sort_keys=True))
PY
chmod -R a-w "$OUTPUT"
