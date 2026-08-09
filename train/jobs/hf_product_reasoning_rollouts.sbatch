#!/bin/bash
#SBATCH -J shohin-product-rollouts
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/hf_product_rollouts_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
RUNTIME=${RUNTIME:?set immutable runtime}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL_ROOT=${MODEL_ROOT:?set pinned model root}
MODEL_REVISION=${MODEL_REVISION:?set pinned model revision}
ADAPTER_CHECKPOINT=${ADAPTER_CHECKPOINT:-}
DATA=${DATA:?set fresh rollout bank}
CANDIDATES_OUTPUT=${CANDIDATES_OUTPUT:?set candidate output}
POSITIVES_OUTPUT=${POSITIVES_OUTPUT:?set positive output}
REPORT=${REPORT:?set report output}
SKIP=${SKIP:-0}
COUNT=${COUNT:-128}
SAMPLES=${SAMPLES:-4}
GENERATION_MODE=${GENERATION_MODE:-qwen-thinking}
PROMPT_BATCH_SIZE=${PROMPT_BATCH_SIZE:-1}
SEED=${SEED:-20260803}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1536}
STAGE_MODEL=${STAGE_MODEL:-true}
FINALIZE_EXHAUSTED=${FINALIZE_EXHAUSTED:-false}
FINALIZE_MAX_NEW_TOKENS=${FINALIZE_MAX_NEW_TOKENS:-64}
FINALIZE_BATCH_SIZE=${FINALIZE_BATCH_SIZE:-32}
ENABLE_THINKING=${ENABLE_THINKING:-true}
BARE_PROMPT_STYLE=${BARE_PROMPT_STYLE:-reasoning}
CODE_TIMEOUT=${CODE_TIMEOUT:-3}
QUANTIZATION=${QUANTIZATION:-none}
BNB_ROOT=${BNB_ROOT:-}
BNB_MANIFEST_SHA256=${BNB_MANIFEST_SHA256:-}
FAST_KERNEL_ROOT=${FAST_KERNEL_ROOT:-}
FAST_KERNEL_MANIFEST_SHA256=${FAST_KERNEL_MANIFEST_SHA256:-}
MODEL_CONFIG_SHA256=${MODEL_CONFIG_SHA256:-}
MODEL_MANIFEST_SHA256=${MODEL_MANIFEST_SHA256:-}

test -x "$PYTHON"
for path in "$RUNTIME/train/hf_product_reasoning_rollouts.py" "$DATA"; do
  [ -f "$path" ] || { echo "[product-rollout] missing input: $path" >&2; exit 2; }
done
if [[ -n "$MODEL_CONFIG_SHA256" ]]; then
  test "$(sha256sum "$MODEL_ROOT/config.json" | cut -d' ' -f1)" = \
    "$MODEL_CONFIG_SHA256"
fi
if [[ -n "$MODEL_MANIFEST_SHA256" ]]; then
  test "$(sha256sum "$MODEL_ROOT/SHA256SUMS" | cut -d' ' -f1)" = \
    "$MODEL_MANIFEST_SHA256"
  (cd "$MODEL_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
fi
if [[ "$QUANTIZATION" == "nf4" ]]; then
  test -n "$BNB_ROOT"; test -n "$BNB_MANIFEST_SHA256"
  test "$(sha256sum "$BNB_ROOT/SHA256SUMS" | cut -d' ' -f1)" = \
    "$BNB_MANIFEST_SHA256"
  (cd "$BNB_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
elif [[ "$QUANTIZATION" != "none" ]]; then
  echo "QUANTIZATION must be none or nf4" >&2
  exit 7
fi
if [[ -n "$FAST_KERNEL_ROOT" ]]; then
  test -n "$FAST_KERNEL_MANIFEST_SHA256"
  test "$(sha256sum "$FAST_KERNEL_ROOT/SHA256SUMS" | cut -d' ' -f1)" = \
    "$FAST_KERNEL_MANIFEST_SHA256"
  (cd "$FAST_KERNEL_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
elif [[ -n "$FAST_KERNEL_MANIFEST_SHA256" ]]; then
  echo "FAST_KERNEL_ROOT is required with its manifest" >&2
  exit 8
fi
if [[ -n "$ADAPTER_CHECKPOINT" ]]; then
  [ -f "$ADAPTER_CHECKPOINT" ] || {
    echo "[product-rollout] missing adapter: $ADAPTER_CHECKPOINT" >&2
    exit 2
  }
fi
for path in "$CANDIDATES_OUTPUT" "$POSITIVES_OUTPUT" "$REPORT"; do
  [ ! -e "$path" ] || { echo "[product-rollout] refusing existing output: $path" >&2; exit 3; }
  mkdir -p "$(dirname "$path")"
done

SOURCE_MODEL_ROOT=$MODEL_ROOT
EFFECTIVE_MODEL_ROOT=$MODEL_ROOT
if [[ "$STAGE_MODEL" == "true" ]]; then
  if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    EFFECTIVE_MODEL_ROOT=$SLURM_TMPDIR/model
    mkdir -p "$EFFECTIVE_MODEL_ROOT"
    cp -a "$SOURCE_MODEL_ROOT"/. "$EFFECTIVE_MODEL_ROOT"/
    test -f "$EFFECTIVE_MODEL_ROOT/config.json"
  else
    echo "[product-rollout] Slurm scratch absent; reading immutable model from Lustre"
  fi
elif [[ "$STAGE_MODEL" != "false" ]]; then
  echo "STAGE_MODEL must be true or false" >&2
  exit 4
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON_PATHS=("$RUNTIME/train" "$RUNTIME/pipeline")
if [[ -n "$BNB_ROOT" ]]; then PYTHON_PATHS=("$BNB_ROOT" "${PYTHON_PATHS[@]}"); fi
if [[ -n "$FAST_KERNEL_ROOT" ]]; then
  PYTHON_PATHS=("$FAST_KERNEL_ROOT" "${PYTHON_PATHS[@]}")
fi
export PYTHONPATH="$(IFS=:; echo "${PYTHON_PATHS[*]}")"
EXTRA_ARGS=()
if [[ -n "$ADAPTER_CHECKPOINT" ]]; then
  EXTRA_ARGS+=(--adapter-checkpoint "$ADAPTER_CHECKPOINT")
fi
if [[ "$FINALIZE_EXHAUSTED" == "true" ]]; then
  EXTRA_ARGS+=(--finalize-exhausted)
elif [[ "$FINALIZE_EXHAUSTED" != "false" ]]; then
  echo "FINALIZE_EXHAUSTED must be true or false" >&2
  exit 5
fi
if [[ "$ENABLE_THINKING" == "true" ]]; then
  EXTRA_ARGS+=(--enable-thinking)
elif [[ "$ENABLE_THINKING" != "false" ]]; then
  echo "ENABLE_THINKING must be true or false" >&2
  exit 6
fi
cd "$RUNTIME/train"
"$PYTHON" hf_product_reasoning_rollouts.py \
  --model-root "$EFFECTIVE_MODEL_ROOT" \
  --model-source-root "$SOURCE_MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --quantization "$QUANTIZATION" \
  --data "$DATA" \
  --candidates-output "$CANDIDATES_OUTPUT" \
  --positives-output "$POSITIVES_OUTPUT" \
  --report "$REPORT" \
  --skip "$SKIP" \
  --count "$COUNT" \
  --samples "$SAMPLES" \
  --generation-mode "$GENERATION_MODE" \
  --prompt-batch-size "$PROMPT_BATCH_SIZE" \
  --finalize-max-new-tokens "$FINALIZE_MAX_NEW_TOKENS" \
  --finalize-batch-size "$FINALIZE_BATCH_SIZE" \
  --code-timeout "$CODE_TIMEOUT" \
  --seed "$SEED" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --bare-prompt-style "$BARE_PROMPT_STYLE" \
  "${EXTRA_ARGS[@]}"
