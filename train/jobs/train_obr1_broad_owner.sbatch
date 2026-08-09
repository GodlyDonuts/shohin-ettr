#!/bin/bash
#SBATCH --job-name=obr1-fit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/obr1_fit_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/obr1_fit_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION MODEL_CONFIG_SHA256 DATA DATA_SHA256 DATA_REPORT DATA_REPORT_SHA256 OUTPUT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
UPDATES=${UPDATES:-2048}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-256}
test ! -e "$OUTPUT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$DATA" | cut -d' ' -f1)" = "$DATA_SHA256"
test "$(sha256sum "$DATA_REPORT" | cut -d' ' -f1)" = "$DATA_REPORT_SHA256"
"$PYTHON" - "$DATA" "$DATA_REPORT" <<'PY'
import hashlib,json,sys
from pathlib import Path
data,report_path=map(Path,sys.argv[1:])
r=json.loads(report_path.read_text()); e=r.get("outputs",{}).get("train",{})
assert r.get("schema")=="shohin-obr1-broad-owner-data-report-v1" and r.get("status")=="complete"
assert r.get("holdout_used") is False and r.get("complete_retention") is True
assert r.get("zero_exact_development_overlap") is True and r.get("zero_ngram_development_overlap") is True
assert r.get("max_sequence_length")==4096 and r.get("admitted_rows",0)>=40000
assert r.get("charged_target_tokens",0)>=12000000
assert Path(e.get("path","")).resolve()==data.resolve()
assert e.get("sha256")==hashlib.sha256(data.read_bytes()).hexdigest()
PY

STAGE_MODEL=${STAGE_MODEL:-false}
if [[ "$STAGE_MODEL" == "true" ]]; then
  LOCAL_PARENT=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
  EFFECTIVE_MODEL_ROOT=$LOCAL_PARENT/model
  mkdir -p "$EFFECTIVE_MODEL_ROOT"
  cp -a "$MODEL_ROOT"/. "$EFFECTIVE_MODEL_ROOT"/
elif [[ "$STAGE_MODEL" == "false" ]]; then
  EFFECTIVE_MODEL_ROOT=$MODEL_ROOT
else
  echo "STAGE_MODEL must be true or false" >&2
  exit 2
fi
mkdir -p logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
nvidia-smi
"$PYTHON" "$RUNTIME/train/train_ecr1_product.py" \
  --model-root "$EFFECTIVE_MODEL_ROOT" --model-source-root "$MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" --model-loader causal \
  --data "$DATA" --output "$OUTPUT" --architecture rme1 --mode shared \
  --controlled-layers 16 --rank 18 --alpha 18 \
  --draft-control draft_unavailable --updates "$UPDATES" \
  --batch-size 1 --gradient-accumulation 8 --max-rows 100000 \
  --max-sequence-length 4096 --learning-rate 2e-5 \
  --seed 2026080911 --data-seed 2026080912 --log-interval 32 \
  --checkpoint-interval "$CHECKPOINT_INTERVAL"
test -f "$OUTPUT/checkpoint_$(printf '%07d' "$UPDATES").pt"
test -f "$OUTPUT/report.json"
chmod -R a-w "$OUTPUT"
