#!/bin/bash
#SBATCH --job-name=ettr-opcode-id
#SBATCH --partition=general
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH --output=/home/%u/shohin/logs/ettr-opcode-id-%j.out
#SBATCH --error=/home/%u/shohin/logs/ettr-opcode-id-%j.err

set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set immutable source root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set exact source commit}
RUNTIME_SHA256SUMS_SHA256=${RUNTIME_SHA256SUMS_SHA256:?set runtime receipt hash}
DATA_ROOT=${DATA_ROOT:?set immutable ETTR data root}
TOKENIZER=${TOKENIZER:?set tokenizer artifact}
OUTPUT=${OUTPUT:?set fresh report path}
WORKERS=${WORKERS:-48}
PYTHON=${PYTHON:-python3}

for path in "$CODE_ROOT" "$DATA_ROOT" "$TOKENIZER" "$OUTPUT"; do
  [[ "$path" == /* ]] || {
    echo "public opcode audit paths must be absolute" >&2
    exit 2
  }
done
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || {
  echo "public opcode audit workers differ" >&2
  exit 2
}
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || {
  echo "public opcode audit output already exists" >&2
  exit 2
}
cd "$CODE_ROOT"
[[ "$(tr -d '\r\n' < SOURCE_COMMIT)" == "$SOURCE_COMMIT" ]] \
  && [[ "$(sha256sum SHA256SUMS | cut -d ' ' -f 1)" \
    == "$RUNTIME_SHA256SUMS_SHA256" ]] \
  && [[ -z "$(sha256sum -c SHA256SUMS 2>&1 | grep -v ': OK$')" ]] \
  || {
    echo "public opcode audit runtime differs" >&2
    exit 2
  }

"$PYTHON" pipeline/audit_ettr_public_opcode_identifiability.py \
  --data-root "$DATA_ROOT" \
  --tokenizer "$TOKENIZER" \
  --output "$OUTPUT" \
  --workers "$WORKERS"

sha256sum "$OUTPUT"
