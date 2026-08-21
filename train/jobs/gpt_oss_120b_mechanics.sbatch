#!/bin/bash
#SBATCH --job-name=gpt-oss-120b-mechanics
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc38,evc50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/gpt_oss_120b_mechanics_%j.out
#SBATCH --error=logs/gpt_oss_120b_mechanics_%j.err

set -euo pipefail

for variable in PYTHON RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT OVERLAY_ROOT OVERLAY_REPORT OUTPUT; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
[[ -x "$PYTHON" ]]
[[ -d "$RUNTIME" && ! -L "$RUNTIME" ]]
[[ -d "$MODEL_ROOT" && ! -L "$MODEL_ROOT" ]]
[[ -d "$OVERLAY_ROOT" && ! -L "$OVERLAY_ROOT" ]]
[[ -f "$OVERLAY_REPORT" && ! -L "$OVERLAY_REPORT" ]]
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]]
[[ "${SLURM_GPUS_ON_NODE:-}" == "1" ]]

source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
q36_verify_runtime
mkdir -p logs "$(dirname "$OUTPUT")"

overlay_manifest_sha256=$("$PYTHON" -P -s -B - "$OVERLAY_REPORT" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("schema") != "shohin-gpt-oss-mxfp4-overlay-v1" or payload.get("status") != "complete":
    raise SystemExit("GPT-OSS overlay report differs")
value = payload.get("manifest_sha256")
if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
    raise SystemExit("GPT-OSS overlay manifest authorization differs")
print(value)
PY
)

export PATH=/apps/slurm/current/bin:/usr/bin:/bin
export PYTHONPATH="$OVERLAY_ROOT:$RUNTIME/train:$RUNTIME/pipeline"
export LOCAL_KERNELS="kernels-community/gpt-oss-triton-kernels=$OVERLAY_ROOT/kernel-repo"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$SLURM_TMPDIR/hf" XDG_CACHE_HOME="$SLURM_TMPDIR/cache"
export TRITON_CACHE_DIR="$SLURM_TMPDIR/triton" TMPDIR="$SLURM_TMPDIR"
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

nvidia-smi
"$PYTHON" -P -s -B "$RUNTIME/train/hf_gpt_oss_120b_mechanics.py" \
  --model-root "$MODEL_ROOT" \
  --model-manifest "$MODEL_ROOT/SHA256SUMS" \
  --expected-model-manifest-sha256 a2a8026b3ab645285045dab439934416628e7bce392b351499b023298ff4a6d0 \
  --overlay-root "$OVERLAY_ROOT" \
  --overlay-manifest "$OVERLAY_ROOT/SHA256SUMS" \
  --expected-overlay-manifest-sha256 "$overlay_manifest_sha256" \
  --output "$OUTPUT"

checkpoint=${OUTPUT%.json}.checkpoint.pt
[[ -f "$OUTPUT" && ! -L "$OUTPUT" ]]
[[ -f "$checkpoint" && ! -L "$checkpoint" ]]
chmod a-w "$OUTPUT" "$checkpoint"
