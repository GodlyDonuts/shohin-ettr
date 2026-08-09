#!/bin/bash
#SBATCH --job-name=mpr2-fit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/mpr2_fit_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/mpr2_fit_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION MODEL_CONFIG_SHA256 DATA DATA_SHA256 DATA_REPORT DATA_REPORT_SHA256 OUTPUT ARM DRAFT_CONTROL; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
case "$ARM:$DRAFT_CONTROL" in aligned:normal|shuffled:normal|hidden:draft_unavailable) ;; *) exit 3 ;; esac
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
UPDATES=${UPDATES:-256}; CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-256}
test ! -e "$OUTPUT"
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"
test "$(sha256sum "$DATA" | cut -d' ' -f1)" = "$DATA_SHA256"
test "$(sha256sum "$DATA_REPORT" | cut -d' ' -f1)" = "$DATA_REPORT_SHA256"
"$PYTHON" - "$DATA" "$DATA_REPORT" "$ARM" <<'PY'
import hashlib,json,sys
from pathlib import Path
data,report_path,arm=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
r=json.loads(report_path.read_text()); key="train_shuffled" if arm=="shuffled" else "train_aligned"; e=r.get("outputs",{}).get(key,{})
assert r.get("schema")=="shohin-mpr2-revision-data-report-v1" and r.get("status")=="complete"
assert r.get("complete_retention") is True and r.get("holdout_used") is False and r.get("max_sequence_length")==4096
assert r.get("trained_olmoe_owned_drafts") is True and r.get("target_multiset_exactly_matched") is True and r.get("zero_source_donor_identity_matches") is True
assert Path(e.get("path","")).resolve()==data.resolve() and e.get("sha256")==hashlib.sha256(data.read_bytes()).hexdigest()
PY
LOCAL_PARENT=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}; EFFECTIVE_MODEL_ROOT=$LOCAL_PARENT/model
mkdir -p "$EFFECTIVE_MODEL_ROOT" logs; cp -a "$MODEL_ROOT"/. "$EFFECTIVE_MODEL_ROOT"/
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
nvidia-smi
"$PYTHON" "$RUNTIME/train/train_ecr1_product.py" --model-root "$EFFECTIVE_MODEL_ROOT" --model-source-root "$MODEL_ROOT" --model-revision "$MODEL_REVISION" --model-loader causal --data "$DATA" --output "$OUTPUT" --architecture rme1 --mode shared --controlled-layers 16 --rank 18 --alpha 18 --draft-control "$DRAFT_CONTROL" --updates "$UPDATES" --batch-size 1 --gradient-accumulation 8 --max-rows 100000 --max-sequence-length 4096 --learning-rate 2e-5 --seed 2026080901 --data-seed 2026080814 --log-interval 8 --checkpoint-interval "$CHECKPOINT_INTERVAL"
test -f "$OUTPUT/checkpoint_$(printf '%07d' "$UPDATES").pt" && test -f "$OUTPUT/report.json"
chmod -R a-w "$OUTPUT"

