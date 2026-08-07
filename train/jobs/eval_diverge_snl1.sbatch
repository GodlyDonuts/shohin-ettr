#!/bin/bash
#SBATCH -J snl1-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 00:45:00
#SBATCH --exclude=evc33,evc43
#SBATCH -o /lustre/fs1/home/%u/shohin/logs/diverge_snl1_eval_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable SNL1 code root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set frozen SNL1 source commit}
OQB1_ROOT=${OQB1_ROOT:?set qualified OQB1 result root}
OQB1_CHECKPOINT_SHA256=${OQB1_CHECKPOINT_SHA256:?set OQB1 checkpoint hash}
OQB1_RESULT_SHA256=${OQB1_RESULT_SHA256:?set OQB1 development hash}
NCP1_ROOT=${NCP1_ROOT:?set qualified NCP1 result root}
NCP1_CHECKPOINT_SHA256=${NCP1_CHECKPOINT_SHA256:?set NCP1 checkpoint hash}
NCP1_RESULT_SHA256=${NCP1_RESULT_SHA256:?set NCP1 development hash}
SVE1_ROOT=${SVE1_ROOT:?set confirmed SVE1 result root}
SVE1_TREATMENT_CHECKPOINT_SHA256=${SVE1_TREATMENT_CHECKPOINT_SHA256:?set SVE1 treatment checkpoint hash}
SVE1_TREATMENT_REPORT_SHA256=${SVE1_TREATMENT_REPORT_SHA256:?set SVE1 treatment report hash}
SVE1_CONTROL_CHECKPOINT_SHA256=${SVE1_CONTROL_CHECKPOINT_SHA256:?set SVE1 control checkpoint hash}
SVE1_CONTROL_REPORT_SHA256=${SVE1_CONTROL_REPORT_SHA256:?set SVE1 control report hash}
SVE1_RESULT_SHA256=${SVE1_RESULT_SHA256:?set SVE1 development hash}
NLS1_ROOT=${NLS1_ROOT:?set frozen NLS1 result root}
NLS1_TREATMENT_CHECKPOINT_SHA256=${NLS1_TREATMENT_CHECKPOINT_SHA256:?set NLS1 treatment checkpoint hash}
NLS1_TREATMENT_REPORT_SHA256=${NLS1_TREATMENT_REPORT_SHA256:?set NLS1 treatment report hash}
NLS1_CONTROL_CHECKPOINT_SHA256=${NLS1_CONTROL_CHECKPOINT_SHA256:?set NLS1 control checkpoint hash}
NLS1_CONTROL_REPORT_SHA256=${NLS1_CONTROL_REPORT_SHA256:?set NLS1 control report hash}
NLS1_RESULT_SHA256=${NLS1_RESULT_SHA256:?set preserved NLS1 development hash}
DATA_ROOT=${DATA_ROOT:?set immutable SNL1 data root}
DATA_REPORT_SHA256=${DATA_REPORT_SHA256:?set SNL1 data report hash}
BOARD_LABEL=${BOARD_LABEL:?set development or confirmation label}
PUBLIC_SHA256=${PUBLIC_SHA256:?set SNL1 public hash}
ASSESSOR_SHA256=${ASSESSOR_SHA256:?set SNL1 assessor hash}
OUTPUT=${OUTPUT:?set isolated SNL1 output}
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
  "$OQB1_ROOT/treatment/checkpoint.pt:$OQB1_CHECKPOINT_SHA256" \
  "$OQB1_ROOT/development.json:$OQB1_RESULT_SHA256" \
  "$NCP1_ROOT/treatment/checkpoint.pt:$NCP1_CHECKPOINT_SHA256" \
  "$NCP1_ROOT/development.json:$NCP1_RESULT_SHA256" \
  "$SVE1_ROOT/treatment/checkpoint.pt:$SVE1_TREATMENT_CHECKPOINT_SHA256" \
  "$SVE1_ROOT/treatment/report.json:$SVE1_TREATMENT_REPORT_SHA256" \
  "$SVE1_ROOT/shuffled_targets/checkpoint.pt:$SVE1_CONTROL_CHECKPOINT_SHA256" \
  "$SVE1_ROOT/shuffled_targets/report.json:$SVE1_CONTROL_REPORT_SHA256" \
  "$SVE1_ROOT/development.json:$SVE1_RESULT_SHA256" \
  "$NLS1_ROOT/treatment/checkpoint.pt:$NLS1_TREATMENT_CHECKPOINT_SHA256" \
  "$NLS1_ROOT/treatment/report.json:$NLS1_TREATMENT_REPORT_SHA256" \
  "$NLS1_ROOT/control/checkpoint.pt:$NLS1_CONTROL_CHECKPOINT_SHA256" \
  "$NLS1_ROOT/control/report.json:$NLS1_CONTROL_REPORT_SHA256" \
  "$NLS1_ROOT/development.json:$NLS1_RESULT_SHA256" \
  "$DATA_ROOT/report.json:$DATA_REPORT_SHA256" \
  "$PUBLIC_DATA:$PUBLIC_SHA256" \
  "$ASSESSOR_DATA:$ASSESSOR_SHA256"; do
  path=${spec%%:*}; digest=${spec##*:}
  test "$(sha256sum "$path" | cut -d' ' -f1)" = "$digest"
done
test ! -e "$OUTPUT"
trap 'test ! -e "$OUTPUT" || chmod a-w "$OUTPUT"' EXIT
nvidia-smi -L
"$PYTHON" "$CODE_ROOT/train/test_diverge_snl1.py"
"$PYTHON" "$CODE_ROOT/train/eval_diverge_snl1.py" \
  --oqb1-checkpoint "$OQB1_ROOT/treatment/checkpoint.pt" \
  --oqb1-checkpoint-sha256 "$OQB1_CHECKPOINT_SHA256" \
  --oqb1-development-report "$OQB1_ROOT/development.json" \
  --oqb1-development-report-sha256 "$OQB1_RESULT_SHA256" \
  --ncp1-checkpoint "$NCP1_ROOT/treatment/checkpoint.pt" \
  --ncp1-checkpoint-sha256 "$NCP1_CHECKPOINT_SHA256" \
  --ncp1-development-report "$NCP1_ROOT/development.json" \
  --ncp1-development-report-sha256 "$NCP1_RESULT_SHA256" \
  --sve1-treatment-checkpoint "$SVE1_ROOT/treatment/checkpoint.pt" \
  --sve1-treatment-checkpoint-sha256 "$SVE1_TREATMENT_CHECKPOINT_SHA256" \
  --sve1-treatment-report "$SVE1_ROOT/treatment/report.json" \
  --sve1-treatment-report-sha256 "$SVE1_TREATMENT_REPORT_SHA256" \
  --sve1-control-checkpoint "$SVE1_ROOT/shuffled_targets/checkpoint.pt" \
  --sve1-control-checkpoint-sha256 "$SVE1_CONTROL_CHECKPOINT_SHA256" \
  --sve1-control-report "$SVE1_ROOT/shuffled_targets/report.json" \
  --sve1-control-report-sha256 "$SVE1_CONTROL_REPORT_SHA256" \
  --sve1-development-report "$SVE1_ROOT/development.json" \
  --sve1-development-report-sha256 "$SVE1_RESULT_SHA256" \
  --nls1-treatment-checkpoint "$NLS1_ROOT/treatment/checkpoint.pt" \
  --nls1-treatment-checkpoint-sha256 "$NLS1_TREATMENT_CHECKPOINT_SHA256" \
  --nls1-treatment-report "$NLS1_ROOT/treatment/report.json" \
  --nls1-treatment-report-sha256 "$NLS1_TREATMENT_REPORT_SHA256" \
  --nls1-control-checkpoint "$NLS1_ROOT/control/checkpoint.pt" \
  --nls1-control-checkpoint-sha256 "$NLS1_CONTROL_CHECKPOINT_SHA256" \
  --nls1-control-report "$NLS1_ROOT/control/report.json" \
  --nls1-control-report-sha256 "$NLS1_CONTROL_REPORT_SHA256" \
  --nls1-development-report "$NLS1_ROOT/development.json" \
  --nls1-development-report-sha256 "$NLS1_RESULT_SHA256" \
  --data-report "$DATA_ROOT/report.json" --data-report-sha256 "$DATA_REPORT_SHA256" \
  --board-label "$BOARD_LABEL" --public-data "$PUBLIC_DATA" \
  --public-data-sha256 "$PUBLIC_SHA256" --assessor-data "$ASSESSOR_DATA" \
  --assessor-data-sha256 "$ASSESSOR_SHA256" --output "$OUTPUT" \
  --source-commit "$SOURCE_COMMIT" --batch-size 64
test "$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$OUTPUT")" = pass
echo "[diverge-snl1-eval] completed board=$BOARD_LABEL"
