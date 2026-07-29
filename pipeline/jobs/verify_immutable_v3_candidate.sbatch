#!/bin/bash
# Independently verify one final v3 token corpus and seal it read-only.
# This job performs no network access and grants no semantic admission.
#SBATCH -J shohin-v3-verify
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/v3_verify_%j.out
#SBATCH --open-mode=append

set -euo pipefail
export PATH=/usr/bin:/bin
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
PY=${PY:-$BASE/miniforge3/bin/python}
SOURCE_ROOT=${SOURCE_ROOT:?set the complete immutable verifier runtime}
CORPUS=${CORPUS:?set one final v3 corpus}
SELECTION_CODE=${SELECTION_CODE:?set the exact source selection code}
OUT=${OUT:?set a fresh verification receipt}
VERIFY_CODE=$SOURCE_ROOT/pipeline/verify_tokenized_shards.py
PARTIAL=$OUT.partial-$SLURM_JOB_ID
SEALED=$OUT.sealed-$SLURM_JOB_ID

cleanup() {
  rm -f -- "$PARTIAL" "$SEALED"
}
trap cleanup EXIT

for path in \
  "$SOURCE_ROOT" \
  "$CORPUS" \
  "$SELECTION_CODE" \
  "$OUT" \
  "$VERIFY_CODE" \
  "$PY"; do
  if [[ "$path" != /* ]]; then
    echo "[v3-verify] all paths must be absolute" >&2
    exit 2
  fi
done
if [[ ! -d "$SOURCE_ROOT" || -L "$SOURCE_ROOT" \
  || ! -s "$SOURCE_ROOT/SHA256SUMS" \
  || ! -d "$CORPUS" || -L "$CORPUS" \
  || "$CORPUS" == *".partial"* \
  || ! -f "$SELECTION_CODE" || -L "$SELECTION_CODE" \
  || ! -x "$PY" || ! -f "$VERIFY_CODE" \
  || -e "$OUT" || -L "$OUT" \
  || -e "$PARTIAL" || -L "$PARTIAL" \
  || -e "$SEALED" || -L "$SEALED" ]]; then
  echo "[v3-verify] preflight differs" >&2
  exit 2
fi
cd "$SOURCE_ROOT"
if [[ -n "$(sha256sum -c SHA256SUMS 2>&1 | grep -v ': OK$')" ]]; then
  echo "[v3-verify] source runtime differs" >&2
  exit 2
fi
mkdir -p "$(dirname "$OUT")"
export PYTHONPATH="$SOURCE_ROOT"
"$PY" -m pipeline.verify_tokenized_shards \
  --shard-dir "$CORPUS" \
  --selection-code "$SELECTION_CODE" \
  --require-external-inputs > "$PARTIAL"
"$PY" - "$PARTIAL" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="ascii"))
required = (
    "all_shards_hash_and_token_verified",
    "document_ledger_verified",
    "external_inputs_verified",
)
if (
    report.get("schema") != "shohin-tokenized-shard-verification-v1"
    or any(report.get(name) is not True for name in required)
):
    raise SystemExit("[v3-verify] verification receipt differs")
PY
find "$CORPUS" -type f -exec chmod 0444 {} +
find "$CORPUS" -depth -type d -exec chmod 0555 {} +
if [[ -n "$(find "$CORPUS" -type f -perm /022 -print -quit)" \
  || -n "$(find "$CORPUS" -type d -perm /022 -print -quit)" ]]; then
  echo "[v3-verify] corpus remains writable after sealing" >&2
  exit 2
fi
"$PY" -m pipeline.verify_tokenized_shards \
  --shard-dir "$CORPUS" \
  --selection-code "$SELECTION_CODE" \
  --require-external-inputs > "$SEALED"
cmp -s -- "$PARTIAL" "$SEALED" || {
  echo "[v3-verify] sealed verification receipt differs" >&2
  exit 2
}
[[ ! -e "$OUT" && ! -L "$OUT" ]] || {
  echo "[v3-verify] final receipt appeared during verification" >&2
  exit 2
}
ln "$SEALED" "$OUT"
chmod 0444 "$OUT"
printf '[v3-verify] complete corpus=%s receipt=%s\n' "$CORPUS" "$OUT"
