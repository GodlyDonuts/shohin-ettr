#!/bin/bash
#SBATCH --job-name=gpt-oss-120b-revision
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc38,evc50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/gpt_oss_120b_revision_%j.out
#SBATCH --error=logs/gpt_oss_120b_revision_%j.err

set -euo pipefail

required=(
  PYTHON RUNTIME RUNTIME_MANIFEST_SHA256 MODEL_ROOT OVERLAY_ROOT OVERLAY_REPORT
  MECHANICS_REPORT DATA OUTPUT
)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
[[ -x "$PYTHON" ]]
for path in "$RUNTIME" "$MODEL_ROOT" "$OVERLAY_ROOT"; do
  [[ -d "$path" && ! -L "$path" ]]
done
for path in "$OVERLAY_REPORT" "$MECHANICS_REPORT" "$DATA"; do
  [[ -f "$path" && ! -L "$path" ]]
done
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]]
[[ "${SLURM_GPUS_ON_NODE:-}" == "1" ]]
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
q36_verify_runtime
mkdir -p logs "$(dirname "$OUTPUT")"

overlay_manifest_sha256=$("$PYTHON" -P -s -B - "$OVERLAY_REPORT" <<'PY'
import json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("schema") != "shohin-gpt-oss-mxfp4-overlay-v1" or payload.get("status") != "complete":
    raise SystemExit("GPT-OSS overlay report differs")
print(payload["manifest_sha256"])
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
"$PYTHON" -P -s -B "$RUNTIME/train/hf_gpt_oss_120b_train_revision.py" \
  --model-root "$MODEL_ROOT" \
  --model-manifest "$MODEL_ROOT/SHA256SUMS" \
  --overlay-root "$OVERLAY_ROOT" \
  --overlay-manifest "$OVERLAY_ROOT/SHA256SUMS" \
  --expected-overlay-manifest-sha256 "$overlay_manifest_sha256" \
  --mechanics-report "$MECHANICS_REPORT" \
  --data "$DATA" \
  --output "$OUTPUT"

checkpoint="$OUTPUT/checkpoint_0000256.pt"
report="$OUTPUT/report.json"
[[ -f "$checkpoint" && ! -L "$checkpoint" ]]
[[ -f "$report" && ! -L "$report" ]]
chmod a-w "$checkpoint" "$report" "$OUTPUT"
