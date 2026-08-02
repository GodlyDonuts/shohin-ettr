#!/bin/bash
#SBATCH --job-name=shohin-terminal-eval-route
#SBATCH --partition=normal
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/terminal-eval-route-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/terminal-eval-route-%j.err

set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set immutable evaluation runtime}
SOURCE_COMMIT=${SOURCE_COMMIT:?set exact evaluation source}
RUNTIME_SHA256SUMS_SHA256=${RUNTIME_SHA256SUMS_SHA256:?set runtime receipt hash}
TERMINAL_RUN_DIR=${TERMINAL_RUN_DIR:?set completed terminal run}
EVAL_OUTPUT=${EVAL_OUTPUT:?set completed evaluation output}
ROUTE_OUTPUT=${ROUTE_OUTPUT:?set fresh route receipt}
PYTHON_ROOT=${PYTHON_ROOT:-/lustre/fs1/home/sa305415/shohin/miniforge3}

for path in "$CODE_ROOT" "$TERMINAL_RUN_DIR" "$EVAL_OUTPUT" \
  "$ROUTE_OUTPUT" "$PYTHON_ROOT"; do
  [[ "$path" == /* ]] || {
    echo "terminal evaluation route paths must be absolute" >&2
    exit 2
  }
done
cd "$CODE_ROOT"
[[ "$(tr -d '\r\n' < SOURCE_COMMIT)" == "$SOURCE_COMMIT" ]] \
  && [[ "$(sha256sum SHA256SUMS | cut -d ' ' -f 1)" \
    == "$RUNTIME_SHA256SUMS_SHA256" ]] \
  && [[ -z "$(sha256sum -c SHA256SUMS 2>&1 | grep -v ': OK$')" ]] \
  || {
    echo "terminal evaluation route runtime differs" >&2
    exit 2
  }
(
  cd "$TERMINAL_RUN_DIR"
  sha256sum -c SHA256SUMS
)
(
  cd "$EVAL_OUTPUT"
  sha256sum -c SHA256SUMS
)
[[ ! -e "$ROUTE_OUTPUT" && ! -L "$ROUTE_OUTPUT" ]] || {
  echo "terminal evaluation route output exists" >&2
  exit 2
}
"$PYTHON_ROOT/bin/python" \
  "$CODE_ROOT/train/route_operation_effect_set_result.py" \
  --report "$EVAL_OUTPUT/report.json" \
  --terminal-contract "$TERMINAL_RUN_DIR/pilot-contract.json" \
  --output "$ROUTE_OUTPUT"
chmod 400 "$ROUTE_OUTPUT"
sha256sum "$ROUTE_OUTPUT"
