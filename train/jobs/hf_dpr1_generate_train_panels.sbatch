#!/bin/bash
#SBATCH --job-name=dpr1-train-panel
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/dpr1_train_panel_%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/dpr1_train_panel_%j.err
set -euo pipefail
for v in RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT MODEL_REVISION MODEL_CONFIG_SHA256 OWNER_CHECKPOINT OWNER_CHECKPOINT_SHA256 SOURCE SOURCE_SHA256 SOURCE_REPORT SOURCE_REPORT_SHA256 OUTPUT REPORT SHARD_INDEX SHARD_COUNT; do test -n "${!v:-}" || exit 2; done
BASE=/lustre/fs1/home/sa305415/shohin; PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}; test ! -e "$OUTPUT" && test ! -e "$REPORT"; test "$(sha256sum "$RUNTIME/SHA256SUMS"|cut -d' ' -f1)" = "$RUNTIME_MANIFEST_SHA256"; (cd "$RUNTIME" && sha256sum -c SHA256SUMS >/dev/null); test "$(sha256sum "$MODEL_ROOT/config.json"|cut -d' ' -f1)" = "$MODEL_CONFIG_SHA256"; test "$(sha256sum "$OWNER_CHECKPOINT"|cut -d' ' -f1)" = "$OWNER_CHECKPOINT_SHA256"; test "$(sha256sum "$SOURCE"|cut -d' ' -f1)" = "$SOURCE_SHA256"; test "$(sha256sum "$SOURCE_REPORT"|cut -d' ' -f1)" = "$SOURCE_REPORT_SHA256"
LOCAL=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/model; mkdir -p "$LOCAL" "$(dirname "$OUTPUT")" logs; cp -a "$MODEL_ROOT"/. "$LOCAL"/; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$RUNTIME/train:$RUNTIME/pipeline"; nvidia-smi
"$PYTHON" "$RUNTIME/train/hf_dpr1_generate_train_panels.py" --source "$SOURCE" --source-report "$SOURCE_REPORT" --model-root "$LOCAL" --model-revision "$MODEL_REVISION" --model-loader causal --owner-checkpoint "$OWNER_CHECKPOINT" --output "$OUTPUT" --report "$REPORT" --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT" --max-new-tokens 768 --seed 2026080925; chmod a-w "$OUTPUT" "$REPORT"

