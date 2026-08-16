#!/bin/bash
#SBATCH --job-name=q36-mtr-mechanics-qualification
#SBATCH --partition=normal
#SBATCH --exclude=evc26,evc29,evc31,evc32,evc33,evc34,evc37,evc38,evc43,evc46,evc50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=logs/q36_mtr_mechanics_qualification_%j.out
#SBATCH --error=logs/q36_mtr_mechanics_qualification_%j.err

set -euo pipefail
for variable in RUNTIME RUNTIME_MANIFEST_SHA256 SOURCE_COMMIT PYTHON RUN_ID DATA DATA_SHA256 OUTPUT ENVIRONMENT_RECEIPT ENVIRONMENT_RECEIPT_SHA256 ENVIRONMENT_TREE_SHA256 QUALIFICATION_AUTHORIZATION QUALIFICATION_AUTHORIZATION_SHA256 PRIOR_TERMINAL_FIRST PRIOR_TERMINAL_SECOND MODEL_MANIFEST; do
  [[ -n "${!variable:-}" ]] || { printf '%s is required\n' "$variable" >&2; exit 2; }
done
source "$RUNTIME/train/jobs/q36_mtr_common.sh"
q36_init_local_tmp
trap q36_cleanup_local_tmp EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
q36_verify_runtime
q36_export_pythonpath
"$PYTHON" -P -s -B \
  "$RUNTIME/pipeline/authorize_q36_mtr_mechanics_qualification.py" verify \
  --authorization "$QUALIFICATION_AUTHORIZATION" \
  --authorization-sha256 "$QUALIFICATION_AUTHORIZATION_SHA256" \
  --source-commit "$SOURCE_COMMIT" \
  --run-id "$RUN_ID" \
  --output-root "$OUTPUT" \
  --runtime-manifest "$RUNTIME/SHA256SUMS" \
  --model-manifest "$MODEL_MANIFEST" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --b1 "$DATA" \
  --prior-terminal "$PRIOR_TERMINAL_FIRST" \
  --prior-terminal "$PRIOR_TERMINAL_SECOND"
q36_verify_model
[[ "$DATA_SHA256" == 2461d6f70b44a142854d56c24e1fb42d600065e5788a2c4e055ba47b12696549 ]] || \
  q36_die "mechanics qualification B1 source differs"
q36_verify_sha256 "$DATA" "$DATA_SHA256"
q36_verify_environment
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || q36_die "mechanics qualification output exists"
model=$(q36_stage_model)
mkdir -p "$(dirname "$OUTPUT")" logs
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
nvidia-smi
"$PYTHON" "$RUNTIME/train/hf_q36_mtr_mechanics.py" \
  --model-root "$model" \
  --model-source-root "$MODEL_ROOT" \
  --model-revision "$MODEL_REVISION" \
  --model-config-sha256 "$MODEL_CONFIG_SHA256" \
  --quantization nf4 \
  --data "$DATA" \
  --output "$OUTPUT" \
  --environment-receipt "$ENVIRONMENT_RECEIPT" \
  --environment-receipt-sha256 "$ENVIRONMENT_RECEIPT_SHA256" \
  --environment-tree-sha256 "$ENVIRONMENT_TREE_SHA256" \
  --rows 24 --seed 2026080825 --data-seed 2026080824
test -f "$OUTPUT/checkpoint_0000001.pt" && test -f "$OUTPUT/report.json"
chmod -R a-w "$OUTPUT"
