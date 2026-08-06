#!/bin/bash
#SBATCH -J eval-rsm1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=256G
#SBATCH -t 03:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/eval_diverge_rsm1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
SOURCE_COMMIT=${SOURCE_COMMIT:?set immutable SOURCE_COMMIT}
TRAIN_OUTPUT=${TRAIN_OUTPUT:?set completed RSM1 training output}
OUTPUT=${OUTPUT:?set isolated evaluation JSON}
TRACE_KIND=${TRACE_KIND:-wrong}
SELECTION_MODE=${SELECTION_MODE:-forced}
ABLATION=${ABLATION:-normal}
DATA_SHA256=${DATA_SHA256:?set verified CRP1 evaluation-board SHA256}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL=$BASE/artifacts/external/smollm3-3b-a07cc9a
SOURCE_CHECKPOINT=$BASE/artifacts/product_reasoning/smollm3-3b-a07cc9a/baseline_late2_v12u1000_rollout_replay_u400_r1/checkpoint_0000400.pt
CRP_CHECKPOINT=$BASE/artifacts/reasoning/diverge_crp1/train_guarded_scientific_43d81d0/checkpoint_0000200.pt
DATA=${DATA:-$BASE/artifacts/reasoning/diverge_crp1/evaluation_seed2026080604.jsonl}

export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" logs "$(dirname "$OUTPUT")"
cd "$BASE"

test -f "$TRAIN_OUTPUT/report.json"
readarray -t RECEIPT < <("$PYTHON" - "$TRAIN_OUTPUT/report.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("schema") != "shohin-diverge-rsm1-training-report-v1":
    raise SystemExit("RSM1 training report schema differs")
print(report["checkpoint"])
print(report["checkpoint_sha256"])
PY
)
RSM_CHECKPOINT=${RECEIPT[0]}
RSM_CHECKPOINT_SHA256=${RECEIPT[1]}
test "$(sha256sum "$RSM_CHECKPOINT" | awk '{print $1}')" = "$RSM_CHECKPOINT_SHA256"
test "$(sha256sum "$DATA" | awk '{print $1}')" = "$DATA_SHA256"
test ! -e "$OUTPUT" || { echo "refusing existing output: $OUTPUT" >&2; exit 2; }
trap 'test ! -e "$OUTPUT" || chmod a-w "$OUTPUT"' EXIT

echo "[eval-rsm1] node=${SLURMD_NODENAME:-?} code=$CODE_ROOT commit=$SOURCE_COMMIT trace=$TRACE_KIND selection=$SELECTION_MODE ablation=$ABLATION"
nvidia-smi -L
"$PYTHON" -m py_compile \
  "$CODE_ROOT/train/diverge_rsm1_data.py" \
  "$CODE_ROOT/train/diverge_rsm1_workspace.py" \
  "$CODE_ROOT/train/diverge_rsm1_product.py" \
  "$CODE_ROOT/train/eval_diverge_rsm1.py"
"$PYTHON" "$CODE_ROOT/train/eval_diverge_rsm1.py" \
  --model-root "$MODEL" \
  --model-revision a07cc9a04f16550a088caea529712d1d335b0ac1 \
  --model-loader causal \
  --source-checkpoint "$SOURCE_CHECKPOINT" \
  --source-checkpoint-sha256 34c82454e0c53609bc1ac6a9f127437080e431f147e28ae63b4080c413d9a82e \
  --crp-checkpoint "$CRP_CHECKPOINT" \
  --crp-checkpoint-sha256 588dce4f608fde47516a8b29feedc40bf7ee58d2ff2aa8b344848915dcacb5ce \
  --rsm-checkpoint "$RSM_CHECKPOINT" \
  --rsm-checkpoint-sha256 "$RSM_CHECKPOINT_SHA256" \
  --expected-update 1600 \
  --data "$DATA" --data-sha256 "$DATA_SHA256" \
  --output "$OUTPUT" --packet-arm guarded \
  --trace-kind "$TRACE_KIND" --selection-mode "$SELECTION_MODE" \
  --ablation "$ABLATION" --batch-size 8 \
  --state-width 256 --state-slots 24 --packet-slots 6 \
  --max-trace-steps 12 --attention-heads 8 --ff-multiplier 4 \
  --max-sequence-length 4096

echo "[eval-rsm1] completed output=$OUTPUT"
