#!/bin/bash
# Build one deterministic, recursively measured ETTR claim runtime.
# This is CPU-only packaging and cannot read training data or checkpoints.
#SBATCH -J shohin-ettr-runtime
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/ettr_runtime_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
ENV_ROOT=${ENV_ROOT:-$BASE/miniforge3}
SOURCE_ROOT=${SOURCE_ROOT:-$BASE}
SOURCE_COMMIT=${SOURCE_COMMIT:?set SOURCE_COMMIT to the reviewed 40-hex commit}
OUT=${OUT:-$BASE/artifacts/runtime/ettr-claim-runtime-v1.tar}
STAGING_PARENT=${SLURM_TMPDIR:-${TMPDIR:-/tmp}}

[[ $SOURCE_COMMIT =~ ^[0-9a-f]{40}$ ]] || {
  echo "[ettr-runtime] SOURCE_COMMIT must be exact 40-hex" >&2
  exit 2
}
[ -d "$SOURCE_ROOT/.git" ] || {
  echo "[ettr-runtime] source repository unavailable" >&2
  exit 3
}
[ -x "$ENV_ROOT/bin/python" ] || {
  echo "[ettr-runtime] source interpreter unavailable" >&2
  exit 4
}
git -C "$SOURCE_ROOT" cat-file -e "$SOURCE_COMMIT^{commit}"
git -C "$SOURCE_ROOT" diff --quiet
git -C "$SOURCE_ROOT" diff --cached --quiet
[ ! -e "$OUT" ] && [ ! -e "$OUT.sha256" ] \
  && [ ! -e "$OUT.inventory.json" ] \
  && [ ! -e "$OUT.runtime-bundle-world.json" ] \
  && [ ! -e "$OUT.runtime-bundle-command.json" ] \
  && [ ! -e "$OUT.runtime-bundle-query.json" ] \
  && [ ! -e "$OUT.stage-policies.json" ] || {
  echo "[ettr-runtime] refusing an existing output" >&2
  exit 5
}

"$ENV_ROOT/bin/python" -I - <<'PY'
import pathlib

import safetensors
import torch

prefix = pathlib.Path(__import__("sys").prefix).resolve()
for name, module in (
    ("safetensors", safetensors),
    ("torch", torch),
):
    origin = pathlib.Path(module.__file__).resolve()
    if not origin.is_relative_to(prefix):
        raise SystemExit(f"[ettr-runtime] {name} escapes source runtime: {origin}")
print(
    "[ettr-runtime] source imports pass "
    f"torch={torch.__version__} safetensors={safetensors.__version__}",
    flush=True,
)
PY

STAGING=$(mktemp -d "$STAGING_PARENT/shohin_ettr_runtime.XXXXXX")
cleanup() {
  chmod -R u+w "$STAGING" 2>/dev/null || true
  rm -rf -- "$STAGING"
}
trap cleanup EXIT
mkdir -p \
  "$STAGING/runtime/miniforge3" \
  "$STAGING/runtime/app/candidate/world" \
  "$STAGING/runtime/app/candidate/command" \
  "$STAGING/runtime/app/candidate/query" \
  "$STAGING/runtime/app/tools" \
  "$STAGING/runtime/app/receipts" \
  "$(dirname "$OUT")"

echo "[ettr-runtime] copying complete interpreter, stdlib, and dependency tree"
cp -a "$ENV_ROOT/bin" "$ENV_ROOT/lib" "$STAGING/runtime/miniforge3/"

COMMON_CANDIDATE_FILES=(
  endogenous_typed_theory_reactor.py
  ettr_factorial_custody.py
  ettr_state_io.py
  model.py
)
declare -A STAGE_RUNNER=(
  [world]=run_ettr_world_compiler.py
  [command]=run_ettr_state_executor.py
  [query]=run_ettr_late_query.py
)
TOOL_FILES=(
  ettr_claim_runtime.py
  ettr_deployment_contract.py
  ettr_runtime_bundle.py
  landlock_stage_exec.py
  run_ettr_verified_stage.py
)
for stage in world command query; do
  for filename in "${COMMON_CANDIDATE_FILES[@]}" "${STAGE_RUNNER[$stage]}"; do
    git -C "$SOURCE_ROOT" show "$SOURCE_COMMIT:train/$filename" \
      > "$STAGING/runtime/app/candidate/$stage/$filename"
  done
done
for filename in "${TOOL_FILES[@]}"; do
  git -C "$SOURCE_ROOT" show "$SOURCE_COMMIT:train/$filename" \
    > "$STAGING/runtime/app/tools/$filename"
done

# Bytecode and .pth execution are neither needed nor admitted.
find "$STAGING/runtime" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$STAGING/runtime" -type f \
  \( -name '*.pyc' -o -name '*.pyo' -o -name '*.pth' \) -delete

for stage in world command query; do
  "$ENV_ROOT/bin/python" -I -B \
    "$STAGING/runtime/app/tools/ettr_runtime_bundle.py" \
    --source-root "$STAGING/runtime/app/candidate/$stage" \
    --receipt \
      "$STAGING/runtime/app/receipts/ettr-runtime-bundle-$stage.json" \
    --stage "$stage"
done
"$ENV_ROOT/bin/python" -I -S -B \
  "$STAGING/runtime/app/tools/ettr_deployment_contract.py" \
  --print-stage-policy-sha256s \
  > "$STAGING/runtime/app/receipts/ettr-stage-policies.json"

# Normalize the exact modes measured by the inventory.
find "$STAGING/runtime" -type d -exec chmod 0555 {} +
find "$STAGING/runtime" -type f -perm /111 -exec chmod 0555 {} +
find "$STAGING/runtime" -type f ! -perm /111 -exec chmod 0444 {} +

echo "[ettr-runtime] building deterministic measured archive"
"$ENV_ROOT/bin/python" -I -S -B \
  "$STAGING/runtime/app/tools/ettr_claim_runtime.py" build \
  --runtime-root "$STAGING/runtime" \
  --source-commit "$SOURCE_COMMIT" \
  --archive "$OUT"
for stage in world command query; do
  cp "$STAGING/runtime/app/receipts/ettr-runtime-bundle-$stage.json" \
    "$OUT.runtime-bundle-$stage.json"
  chmod 0444 "$OUT.runtime-bundle-$stage.json"
done
cp "$STAGING/runtime/app/receipts/ettr-stage-policies.json" \
  "$OUT.stage-policies.json"
chmod 0444 "$OUT.stage-policies.json"

echo "[ettr-runtime] complete"
sha256sum \
  "$OUT" \
  "$OUT.inventory.json" \
  "$OUT.runtime-bundle-world.json" \
  "$OUT.runtime-bundle-command.json" \
  "$OUT.runtime-bundle-query.json" \
  "$OUT.stage-policies.json"
ls -lh \
  "$OUT" \
  "$OUT.inventory.json" \
  "$OUT.runtime-bundle-world.json" \
  "$OUT.runtime-bundle-command.json" \
  "$OUT.runtime-bundle-query.json" \
  "$OUT.stage-policies.json"
