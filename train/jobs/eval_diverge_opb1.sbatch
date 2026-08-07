#!/bin/bash
#SBATCH -J opb1-eval
#SBATCH -p normal
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:45:00
#SBATCH -o /lustre/fs1/home/%u/shohin/logs/diverge_opb1_eval_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable OPB1 code root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set frozen OPB1 source commit}
SNL1_ROOT=${SNL1_ROOT:?set confirmed SNL1 result root}
SNL1_RESULT_SHA256=${SNL1_RESULT_SHA256:?set SNL1 development hash}
SNL1_DATA_REPORT_SHA256=${SNL1_DATA_REPORT_SHA256:?set SNL1 data report hash}
OQB1_CHECKPOINT=${OQB1_CHECKPOINT:?set qualified OQB1 checkpoint}
OQB1_CHECKPOINT_SHA256=${OQB1_CHECKPOINT_SHA256:?set OQB1 checkpoint hash}
NCP1_CHECKPOINT=${NCP1_CHECKPOINT:?set qualified NCP1 checkpoint}
NCP1_CHECKPOINT_SHA256=${NCP1_CHECKPOINT_SHA256:?set NCP1 checkpoint hash}
SVE1_CHECKPOINT=${SVE1_CHECKPOINT:?set qualified SVE1 checkpoint}
SVE1_CHECKPOINT_SHA256=${SVE1_CHECKPOINT_SHA256:?set SVE1 checkpoint hash}
NLS1_CHECKPOINT=${NLS1_CHECKPOINT:?set frozen NLS1 treatment checkpoint}
NLS1_CHECKPOINT_SHA256=${NLS1_CHECKPOINT_SHA256:?set NLS1 checkpoint hash}
TREATMENT_ROOT=${TREATMENT_ROOT:?set OPB1 treatment root}
TREATMENT_CHECKPOINT_SHA256=${TREATMENT_CHECKPOINT_SHA256:?set treatment checkpoint hash}
TREATMENT_REPORT_SHA256=${TREATMENT_REPORT_SHA256:?set treatment report hash}
CONTROL_ROOT=${CONTROL_ROOT:?set OPB1 control root}
CONTROL_CHECKPOINT_SHA256=${CONTROL_CHECKPOINT_SHA256:?set control checkpoint hash}
CONTROL_REPORT_SHA256=${CONTROL_REPORT_SHA256:?set control report hash}
DATA_ROOT=${DATA_ROOT:?set immutable OPB1 data root}
DATA_REPORT_SHA256=${DATA_REPORT_SHA256:?set OPB1 data report hash}
BOARD_LABEL=${BOARD_LABEL:?set development or confirmation seed label}
PUBLIC_SHA256=${PUBLIC_SHA256:?set OPB1 public hash}
ASSESSOR_SHA256=${ASSESSOR_SHA256:?set OPB1 assessor hash}
OUTPUT=${OUTPUT:?set isolated OPB1 evaluation output}
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
PUBLIC_DATA=$DATA_ROOT/${BOARD_LABEL}_public.jsonl
ASSESSOR_DATA=$DATA_ROOT/${BOARD_LABEL}_assessor.jsonl
for spec in \
  "$SNL1_ROOT/development.json:$SNL1_RESULT_SHA256" \
  "$OQB1_CHECKPOINT:$OQB1_CHECKPOINT_SHA256" \
  "$NCP1_CHECKPOINT:$NCP1_CHECKPOINT_SHA256" \
  "$SVE1_CHECKPOINT:$SVE1_CHECKPOINT_SHA256" \
  "$NLS1_CHECKPOINT:$NLS1_CHECKPOINT_SHA256" \
  "$TREATMENT_ROOT/checkpoint.pt:$TREATMENT_CHECKPOINT_SHA256" \
  "$TREATMENT_ROOT/report.json:$TREATMENT_REPORT_SHA256" \
  "$CONTROL_ROOT/checkpoint.pt:$CONTROL_CHECKPOINT_SHA256" \
  "$CONTROL_ROOT/report.json:$CONTROL_REPORT_SHA256" \
  "$DATA_ROOT/report.json:$DATA_REPORT_SHA256" \
  "$PUBLIC_DATA:$PUBLIC_SHA256" \
  "$ASSESSOR_DATA:$ASSESSOR_SHA256"; do
  path=${spec%%:*}; digest=${spec##*:}
  test "$(sha256sum "$path" | cut -d' ' -f1)" = "$digest"
done
test ! -e "$OUTPUT"
trap 'test ! -e "$OUTPUT" || chmod a-w "$OUTPUT"' EXIT
nvidia-smi -L
"$PYTHON" "$CODE_ROOT/train/test_diverge_opb1.py"
"$PYTHON" "$CODE_ROOT/train/eval_diverge_opb1.py" \
  --snl1-development-report "$SNL1_ROOT/development.json" \
  --snl1-development-report-sha256 "$SNL1_RESULT_SHA256" \
  --snl1-data-report-sha256 "$SNL1_DATA_REPORT_SHA256" \
  --oqb1-checkpoint "$OQB1_CHECKPOINT" \
  --oqb1-checkpoint-sha256 "$OQB1_CHECKPOINT_SHA256" \
  --ncp1-checkpoint "$NCP1_CHECKPOINT" \
  --ncp1-checkpoint-sha256 "$NCP1_CHECKPOINT_SHA256" \
  --sve1-checkpoint "$SVE1_CHECKPOINT" \
  --sve1-checkpoint-sha256 "$SVE1_CHECKPOINT_SHA256" \
  --nls1-checkpoint "$NLS1_CHECKPOINT" \
  --nls1-checkpoint-sha256 "$NLS1_CHECKPOINT_SHA256" \
  --treatment-checkpoint "$TREATMENT_ROOT/checkpoint.pt" \
  --treatment-checkpoint-sha256 "$TREATMENT_CHECKPOINT_SHA256" \
  --treatment-report "$TREATMENT_ROOT/report.json" \
  --treatment-report-sha256 "$TREATMENT_REPORT_SHA256" \
  --control-checkpoint "$CONTROL_ROOT/checkpoint.pt" \
  --control-checkpoint-sha256 "$CONTROL_CHECKPOINT_SHA256" \
  --control-report "$CONTROL_ROOT/report.json" \
  --control-report-sha256 "$CONTROL_REPORT_SHA256" \
  --data-report "$DATA_ROOT/report.json" \
  --data-report-sha256 "$DATA_REPORT_SHA256" --board-label "$BOARD_LABEL" \
  --public-data "$PUBLIC_DATA" --public-data-sha256 "$PUBLIC_SHA256" \
  --assessor-data "$ASSESSOR_DATA" --assessor-data-sha256 "$ASSESSOR_SHA256" \
  --output "$OUTPUT" --source-commit "$SOURCE_COMMIT" --batch-size 64
test "$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$OUTPUT")" = pass
echo "[diverge-opb1-eval] completed board=$BOARD_LABEL"
