#!/bin/bash
#SBATCH --job-name=shohin-ettr-compose
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/joint-compose-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/joint-compose-%j.err

set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set immutable source root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set exact source commit}
RUNTIME_SHA256SUMS_SHA256=${RUNTIME_SHA256SUMS_SHA256:?set runtime receipt hash}
PARENT_RUN_DIR=${PARENT_RUN_DIR:?set parent joint run}
COMPILER=${COMPILER:?set compiler component}
COMPILER_SHA256=${COMPILER_SHA256:?set compiler hash}
REACTOR=${REACTOR:?set reactor component}
REACTOR_SHA256=${REACTOR_SHA256:?set reactor hash}
READER=${READER:?set reader component}
READER_SHA256=${READER_SHA256:?set reader hash}
OUTPUT=${OUTPUT:?set fresh composition output}
QUERY_READOUT_GEOMETRY=${QUERY_READOUT_GEOMETRY:-stage}
READER_CONTRACT=${READER_CONTRACT:-}
READER_CONTRACT_SHA256=${READER_CONTRACT_SHA256:-}
READER_REPORT=${READER_REPORT:-}
READER_REPORT_SHA256=${READER_REPORT_SHA256:-}
PYTHON_ROOT=${PYTHON_ROOT:-/lustre/fs1/home/sa305415/shohin/miniforge3}

for path in "$CODE_ROOT" "$PARENT_RUN_DIR" "$COMPILER" "$REACTOR" \
  "$READER" "$OUTPUT" "$PYTHON_ROOT"; do
  [[ "$path" == /* ]] || {
    echo "joint composition paths must be absolute" >&2
    exit 2
  }
done
cd "$CODE_ROOT"
[[ "$(tr -d '\r\n' < SOURCE_COMMIT)" == "$SOURCE_COMMIT" ]] \
  && [[ "$(sha256sum SHA256SUMS | awk '{print $1}')" \
    == "$RUNTIME_SHA256SUMS_SHA256" ]] \
  && [[ -z "$(sha256sum -c SHA256SUMS 2>&1 | grep -v ': OK$')" ]] \
  || {
    echo "joint composition runtime differs" >&2
    exit 2
  }
(
  cd "$PARENT_RUN_DIR"
  sha256sum -c SHA256SUMS
)
printf '%s  %s\n' "$COMPILER_SHA256" "$COMPILER" \
  "$REACTOR_SHA256" "$REACTOR" \
  "$READER_SHA256" "$READER" \
  | sha256sum -c -
if [[ -e "$OUTPUT" || -L "$OUTPUT" ]]; then
  echo "joint composition output exists" >&2
  exit 2
fi

parent_contract_sha=$(sha256sum \
  "$PARENT_RUN_DIR/run-contract.json" | awk '{print $1}')
parent_model_sha=$(sha256sum \
  "$PARENT_RUN_DIR/joint-model-final.pt" | awk '{print $1}')

reader_geometry_args=(
  --query-readout-geometry "$QUERY_READOUT_GEOMETRY"
)
if [[ "$QUERY_READOUT_GEOMETRY" != stage ]]; then
  [[ "$READER_CONTRACT" == /* && "$READER_REPORT" == /* \
    && "$READER_CONTRACT_SHA256" =~ ^[0-9a-f]{64}$ \
    && "$READER_REPORT_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
      echo "non-stage reader receipts differ" >&2
      exit 2
    }
  reader_geometry_args+=(
    --reader-contract "$READER_CONTRACT"
    --reader-contract-sha256 "$READER_CONTRACT_SHA256"
    --reader-report "$READER_REPORT"
    --reader-report-sha256 "$READER_REPORT_SHA256"
  )
elif [[ -n "$READER_CONTRACT$READER_CONTRACT_SHA256$READER_REPORT$READER_REPORT_SHA256" ]]; then
  echo "stage reader must not carry geometry receipts" >&2
  exit 2
fi

"$PYTHON_ROOT/bin/python" \
  "$CODE_ROOT/train/compose_ettr_joint_components.py" \
  --parent-joint-model "$PARENT_RUN_DIR/joint-model-final.pt" \
  --parent-joint-model-sha256 "$parent_model_sha" \
  --parent-run-contract "$PARENT_RUN_DIR/run-contract.json" \
  --parent-run-contract-sha256 "$parent_contract_sha" \
  --compiler "$COMPILER" \
  --compiler-sha256 "$COMPILER_SHA256" \
  --reactor "$REACTOR" \
  --reactor-sha256 "$REACTOR_SHA256" \
  --reader "$READER" \
  --reader-sha256 "$READER_SHA256" \
  --output "$OUTPUT" \
  --source-commit "$SOURCE_COMMIT" \
  "${reader_geometry_args[@]}"

(
  cd "$OUTPUT"
  sha256sum \
    composition-report.json \
    joint-model-final.pt \
    run-contract.json \
    > SHA256SUMS
  chmod 400 SHA256SUMS
  sha256sum -c SHA256SUMS
)
