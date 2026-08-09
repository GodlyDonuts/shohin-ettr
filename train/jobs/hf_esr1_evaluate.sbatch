#!/bin/bash
#SBATCH --job-name=esr1-eval
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/esr1_eval_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/esr1_eval_%j.err

set -euo pipefail

for variable in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION \
  MODEL_CONFIG_SHA256 ADAPTER_CHECKPOINT ADAPTER_CHECKPOINT_SHA256 DATA \
  DATA_SHA256 DATA_REPORT DATA_REPORT_SHA256 CANDIDATES_OUTPUT REPORT \
  SHARD_INDEX SHARD_COUNT; do
  test -n "${!variable:-}" || { echo "$variable is required" >&2; exit 2; }
done
BASE=/lustre/fs1/home/sa305415/shohin
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
test -x "$PYTHON"
for path in "$RUNTIME/train/hf_idr1_evaluate_reviser.py" \
  "$RUNTIME/train/hf_product_reasoning_eval.py" \
  "$RUNTIME/train/hf_product_reasoning_train.py" \
  "$RUNTIME/train/integrated_reasoning_workspace.py" "$MODEL_ROOT/config.json" \
  "$ADAPTER_CHECKPOINT" "$DATA" "$DATA_REPORT"; do test -f "$path"; done
test "$(sha256sum "$RUNTIME/SHA256SUMS" | cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"
(cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
printf '%s  %s\n' "$MODEL_CONFIG_SHA256" "$MODEL_ROOT/config.json" | sha256sum -c - >/dev/null
printf '%s  %s\n' "$ADAPTER_CHECKPOINT_SHA256" "$ADAPTER_CHECKPOINT" | sha256sum -c - >/dev/null
printf '%s  %s\n' "$DATA_SHA256" "$DATA" | sha256sum -c - >/dev/null
printf '%s  %s\n' "$DATA_REPORT_SHA256" "$DATA_REPORT" | sha256sum -c - >/dev/null
test ! -e "$CANDIDATES_OUTPUT"; test ! -e "$REPORT"

export TMPDIR=$BASE/tmp/$SLURM_JOB_ID
trap 'rm -rf "$TMPDIR"' EXIT
mkdir -p "$TMPDIR" logs "$(dirname "$REPORT")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"
nvidia-smi

"$PYTHON" "$RUNTIME/train/hf_idr1_evaluate_reviser.py" \
  --model-root "$MODEL_ROOT" --model-source-root "$MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" --model-loader auto \
  --adapter-checkpoint "$ADAPTER_CHECKPOINT" --data "$DATA" \
  --data-report "$DATA_REPORT" --split development \
  --candidates-output "$CANDIDATES_OUTPUT" --report "$REPORT" \
  --batch-size 2 --max-new-tokens 768 --seed 2026080816 \
  --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT"
sha256sum "$CANDIDATES_OUTPUT" "$REPORT"
chmod a-w "$CANDIDATES_OUTPUT" "$REPORT"
