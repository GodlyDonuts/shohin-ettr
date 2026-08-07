#!/bin/bash
#SBATCH -J nls1-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:20:00
#SBATCH -o /lustre/fs1/home/%u/shohin/logs/diverge_nls1_eval_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable NLS1 code root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set NLS1 source commit}
EAL2_ROOT=${EAL2_ROOT:?set frozen passing EAL2 root}
EAL2_CHECKPOINT_SHA256=${EAL2_CHECKPOINT_SHA256:?set EAL2 checkpoint hash}
EAL2_RESULT_SHA256=${EAL2_RESULT_SHA256:?set EAL2 development result hash}
TREATMENT_ROOT=${TREATMENT_ROOT:?set NLS1 treatment root}
TREATMENT_CHECKPOINT_SHA256=${TREATMENT_CHECKPOINT_SHA256:?set treatment checkpoint hash}
TREATMENT_REPORT_SHA256=${TREATMENT_REPORT_SHA256:?set treatment report hash}
CONTROL_ROOT=${CONTROL_ROOT:?set NLS1 control root}
CONTROL_CHECKPOINT_SHA256=${CONTROL_CHECKPOINT_SHA256:?set control checkpoint hash}
CONTROL_REPORT_SHA256=${CONTROL_REPORT_SHA256:?set control report hash}
PUBLIC_DATA=${PUBLIC_DATA:?set NLS1 public board}
PUBLIC_SHA256=${PUBLIC_SHA256:?set NLS1 public hash}
ASSESSOR_DATA=${ASSESSOR_DATA:?set NLS1 assessor board}
ASSESSOR_SHA256=${ASSESSOR_SHA256:?set NLS1 assessor hash}
OUTPUT=${OUTPUT:?set isolated NLS1 evaluation output}
PARENT_GATE=${PARENT_GATE:-}
PARENT_GATE_SHA256=${PARENT_GATE_SHA256:-}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}

export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs" "$(dirname "$OUTPUT")"
(cd "$CODE_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
if [[ -n "$PARENT_GATE" || -n "$PARENT_GATE_SHA256" ]]; then
  test -n "$PARENT_GATE" && test -n "$PARENT_GATE_SHA256"
  test "$(sha256sum "$PARENT_GATE" | cut -d' ' -f1)" = "$PARENT_GATE_SHA256"
  test "$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$PARENT_GATE")" = pass
fi
for spec in \
  "$EAL2_ROOT/training/checkpoint.pt:$EAL2_CHECKPOINT_SHA256" \
  "$EAL2_ROOT/development.json:$EAL2_RESULT_SHA256" \
  "$TREATMENT_ROOT/checkpoint.pt:$TREATMENT_CHECKPOINT_SHA256" \
  "$TREATMENT_ROOT/report.json:$TREATMENT_REPORT_SHA256" \
  "$CONTROL_ROOT/checkpoint.pt:$CONTROL_CHECKPOINT_SHA256" \
  "$CONTROL_ROOT/report.json:$CONTROL_REPORT_SHA256" \
  "$PUBLIC_DATA:$PUBLIC_SHA256" \
  "$ASSESSOR_DATA:$ASSESSOR_SHA256"; do
  path=${spec%%:*}; digest=${spec##*:}
  test "$(sha256sum "$path" | cut -d' ' -f1)" = "$digest"
done
test ! -e "$OUTPUT"
trap 'test ! -e "$OUTPUT" || chmod a-w "$OUTPUT"' EXIT
nvidia-smi -L
"$PYTHON" "$CODE_ROOT/train/test_diverge_nls1.py"
"$PYTHON" "$CODE_ROOT/train/eval_diverge_nls1.py" \
  --eal2-checkpoint "$EAL2_ROOT/training/checkpoint.pt" \
  --eal2-checkpoint-sha256 "$EAL2_CHECKPOINT_SHA256" \
  --eal2-development-report "$EAL2_ROOT/development.json" \
  --eal2-development-report-sha256 "$EAL2_RESULT_SHA256" \
  --treatment-checkpoint "$TREATMENT_ROOT/checkpoint.pt" \
  --treatment-checkpoint-sha256 "$TREATMENT_CHECKPOINT_SHA256" \
  --treatment-report "$TREATMENT_ROOT/report.json" \
  --treatment-report-sha256 "$TREATMENT_REPORT_SHA256" \
  --control-checkpoint "$CONTROL_ROOT/checkpoint.pt" \
  --control-checkpoint-sha256 "$CONTROL_CHECKPOINT_SHA256" \
  --control-report "$CONTROL_ROOT/report.json" \
  --control-report-sha256 "$CONTROL_REPORT_SHA256" \
  --public-data "$PUBLIC_DATA" --public-data-sha256 "$PUBLIC_SHA256" \
  --assessor-data "$ASSESSOR_DATA" --assessor-data-sha256 "$ASSESSOR_SHA256" \
  --output "$OUTPUT" --source-commit "$SOURCE_COMMIT" --batch-size 512
test "$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$OUTPUT")" = pass
echo "[diverge-nls1-eval] completed"
