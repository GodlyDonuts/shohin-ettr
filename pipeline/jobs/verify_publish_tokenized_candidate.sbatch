#!/bin/bash
#SBATCH -J shohin-verify-publish
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/verify_publish_%j.out
#SBATCH --open-mode=append
# Recover a completed tokenized candidate whose original build failed closed
# during final verification. This never rebuilds or admits training data.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=${PYTHON:-$BASE/miniforge3/bin/python}
CODE_ROOT=${CODE_ROOT:?set CODE_ROOT to the original isolated code root}
SHARD_DIR=${SHARD_DIR:?set SHARD_DIR to the completed .partial candidate}
OUT=${OUT:?set OUT to the fresh final candidate path}
SELECTION_CODE=${SELECTION_CODE:-$CODE_ROOT/pipeline/tokenize_shards.py}
VERIFY_SHARDS=${VERIFY_SHARDS:-$CODE_ROOT/pipeline/verify_tokenized_shards.py}
SELECTION_CODE_SHA256=${SELECTION_CODE_SHA256:?set exact selection-code SHA-256}
VERIFY_SHARDS_SHA256=${VERIFY_SHARDS_SHA256:?set exact verifier SHA-256}
PREWARM_READS=${PREWARM_READS:-0}

for path in "$CODE_ROOT" "$SHARD_DIR" "$OUT" "$SELECTION_CODE" "$VERIFY_SHARDS"; do
  [[ "$path" == /* ]] || {
    echo "[verify-publish] paths must be absolute: $path" >&2
    exit 2
  }
done
[[ "$SHARD_DIR" == "$OUT.partial" ]] || {
  echo "[verify-publish] partial path must be exactly OUT.partial" >&2
  exit 2
}
[[ -d "$CODE_ROOT" && ! -L "$CODE_ROOT" ]] || {
  echo "[verify-publish] isolated code root is unavailable" >&2
  exit 3
}
[[ -d "$SHARD_DIR" && ! -L "$SHARD_DIR" ]] || {
  echo "[verify-publish] partial candidate is unavailable" >&2
  exit 3
}
[[ ! -e "$OUT" && ! -L "$OUT" ]] || {
  echo "[verify-publish] refusing existing final path" >&2
  exit 4
}
[[ -f "$SHARD_DIR/manifest.json" && ! -L "$SHARD_DIR/manifest.json" ]] || {
  echo "[verify-publish] candidate manifest is unavailable" >&2
  exit 3
}
[[ "$PREWARM_READS" == 0 || "$PREWARM_READS" == 1 ]] || {
  echo "[verify-publish] PREWARM_READS must be 0 or 1" >&2
  exit 2
}
printf '%s  %s\n' \
  "$SELECTION_CODE_SHA256" "$SELECTION_CODE" \
  "$VERIFY_SHARDS_SHA256" "$VERIFY_SHARDS" | sha256sum -c -

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
if [[ "$PREWARM_READS" == 1 ]]; then
  # Lustre can settle cache/metadata on a file's first complete read after a
  # writer exits. Warm every regular file, then run the unchanged verifier.
  # This pass grants no trust: the verifier below re-hashes, decompresses, and
  # reconciles every admitted byte after the warmup is complete.
  "$PY" - "$SHARD_DIR" <<'PY'
import os
import stat
import sys
from pathlib import Path

root = Path(sys.argv[1])
for path in sorted(root.iterdir(), key=lambda value: value.name):
    metadata = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise SystemExit(f"[verify-publish] prewarm rejected non-regular file: {path}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        while os.read(descriptor, 8 * 1024 * 1024):
            pass
    finally:
        os.close(descriptor)
PY
  sleep 5
fi
cd "$CODE_ROOT"
PYTHONPATH="$CODE_ROOT" "$PY" -m pipeline.verify_tokenized_shards \
  --shard-dir "$SHARD_DIR" \
  --selection-code "$SELECTION_CODE" \
  --require-external-inputs

[[ ! -e "$OUT" && ! -L "$OUT" ]] || {
  echo "[verify-publish] final path appeared during verification" >&2
  exit 4
}
mv -T -- "$SHARD_DIR" "$OUT"
echo "[verify-publish] verified candidate committed at $OUT; not training-admitted"
