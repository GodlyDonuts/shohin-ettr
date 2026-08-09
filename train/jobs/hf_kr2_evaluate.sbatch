#!/bin/bash
#SBATCH --job-name=kr2-eval
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/kr2_eval_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/kr2_eval_%j.err

set -euo pipefail

for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION \
  MODEL_CONFIG_SHA256 MODEL_MANIFEST_SHA256 TRAIN_OUTPUT WARM_START_SHA256 \
  DATA DATA_SHA256 DATA_REPORT DATA_REPORT_SHA256 CANDIDATES_OUTPUT REPORT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
SHARD_INDEX=${SHARD_INDEX:-0}
SHARD_COUNT=${SHARD_COUNT:-1}
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
CHECKPOINT=$TRAIN_OUTPUT/checkpoint_0000256.pt
TRAIN_REPORT=$TRAIN_OUTPUT/report.json
for path in "$CHECKPOINT" "$TRAIN_REPORT" "$DATA" "$DATA_REPORT"; do
  test -f "$path"
done
test ! -e "$CANDIDATES_OUTPUT"
test ! -e "$REPORT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = \
  "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = \
  "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$MODEL_ROOT/SHA256SUMS" | cut -d' ' -f1)" = \
  "$MODEL_MANIFEST_SHA256"
(cd "$MODEL_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$DATA" | cut -d' ' -f1)" = "$DATA_SHA256"
test "$(sha256sum "$DATA_REPORT" | cut -d' ' -f1)" = "$DATA_REPORT_SHA256"
CHECKPOINT_SHA256=$(sha256sum "$CHECKPOINT" | cut -d' ' -f1)

"$PYTHON" - "$TRAIN_REPORT" "$CHECKPOINT" "$WARM_START_SHA256" <<'PY'
import json, sys, torch
from pathlib import Path
report_path, checkpoint_path = map(Path, sys.argv[1:3])
report = json.loads(report_path.read_text())
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
metadata = checkpoint.get("metadata", {})
if report.get("status") != "complete" or report.get("update") != 256:
    raise SystemExit("KR2 train report is incomplete")
if checkpoint.get("update") != 256:
    raise SystemExit("KR2 checkpoint update differs")
if metadata.get("warm_start_sha256") != sys.argv[3]:
    raise SystemExit("KR2 warm-start lineage differs")
PY

SOURCE_MODEL_ROOT=$MODEL_ROOT
EFFECTIVE_MODEL_ROOT=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/model
mkdir -p "$EFFECTIVE_MODEL_ROOT" "$(dirname "$REPORT")"
cp -a "$SOURCE_MODEL_ROOT"/. "$EFFECTIVE_MODEL_ROOT"/
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
nvidia-smi
"$PYTHON" "$RUNTIME/train/hf_idr1_evaluate_reviser.py" \
  --model-root "$EFFECTIVE_MODEL_ROOT" --model-source-root "$SOURCE_MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" --model-loader multimodal \
  --adapter-checkpoint "$CHECKPOINT" --data "$DATA" --data-report "$DATA_REPORT" \
  --split development --candidates-output "$CANDIDATES_OUTPUT" --report "$REPORT" \
  --batch-size 2 --max-new-tokens 768 --seed 2026080816 \
  --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT"
sha256sum "$CANDIDATES_OUTPUT" "$REPORT"
chmod a-w "$CANDIDATES_OUTPUT" "$REPORT"
