#!/bin/bash
# Build the immutable optimizer-visible ETTR-IL-v3 release only after the
# independent materialization and main/confirmation separation audits pass.
# The job is CPU-only and cannot read or write model checkpoint bytes.
#SBATCH -J shohin-ettr-v3-release
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 2-00:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/ettr_il_v3_release_%j.out
#SBATCH --open-mode=append
set -euo pipefail
export PATH=/usr/bin:/bin
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
PY=${PY:-$BASE/miniforge3/bin/python}
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
SOURCE_ROOT=${SOURCE_ROOT:?set SOURCE_ROOT to a clean exact-source checkout}
SOURCE_COMMIT=${SOURCE_COMMIT:?set SOURCE_COMMIT to the exact 40-hex commit}
MAIN_AUDIT=${MAIN_AUDIT:?set MAIN_AUDIT to the passing main audit}
SEPARATION=${SEPARATION:?set SEPARATION to the passing separation report}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to immutable main materialized data}
TOKENIZER=${TOKENIZER:?set TOKENIZER to the exact corpus tokenizer}
OUT=${OUT:?set OUT to a fresh release directory}
PROTECTED_CHECKPOINT_SHA256=${PROTECTED_CHECKPOINT_SHA256:?set protected checkpoint SHA-256}

[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  echo "[ettr-v3-release] malformed source commit" >&2
  exit 2
}
[[ "$PROTECTED_CHECKPOINT_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  echo "[ettr-v3-release] malformed protected-checkpoint SHA-256" >&2
  exit 2
}
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || {
  echo "[ettr-v3-release] malformed worker count" >&2
  exit 2
}
(( WORKERS <= SLURM_CPUS_PER_TASK )) || {
  echo "[ettr-v3-release] worker count exceeds allocated CPUs" >&2
  exit 2
}
[[ -x "$PY" && -d "$SOURCE_ROOT/.git" && ! -L "$SOURCE_ROOT" ]] || {
  echo "[ettr-v3-release] exact-source checkout or Python is unavailable" >&2
  exit 3
}
[[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || {
  echo "[ettr-v3-release] source HEAD differs" >&2
  exit 3
}
[[ -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" ]] || {
  echo "[ettr-v3-release] source checkout is not clean" >&2
  exit 3
}
for path in "$MAIN_AUDIT" "$SEPARATION" "$TOKENIZER"; do
  [[ -f "$path" && ! -L "$path" ]] || {
    echo "[ettr-v3-release] required file is unavailable: $path" >&2
    exit 4
  }
done
[[ -d "$DATA_ROOT" && ! -L "$DATA_ROOT" ]] || {
  echo "[ettr-v3-release] materialized data root is unavailable" >&2
  exit 4
}
[[ ! -e "$OUT" ]] || {
  echo "[ettr-v3-release] refusing existing output: $OUT" >&2
  exit 5
}

cd "$SOURCE_ROOT"
"$PY" - <<'PY'
import torch
import tokenizers
assert torch.__version__
assert tokenizers.__version__
PY
"$PY" -u pipeline/build_ettr_il_v3_training_release.py \
  --main-audit "$MAIN_AUDIT" \
  --separation-report "$SEPARATION" \
  --data-root "$DATA_ROOT" \
  --tokenizer "$TOKENIZER" \
  --protected-checkpoint-sha256 "$PROTECTED_CHECKPOINT_SHA256" \
  --source-commit "$SOURCE_COMMIT" \
  --output "$OUT" \
  --workers "$WORKERS"

[[ -f "$OUT/release.json" && ! -L "$OUT/release.json" ]] || {
  echo "[ettr-v3-release] release artifact is unavailable" >&2
  exit 6
}
"$PY" - "$OUT/release.json" "$SOURCE_COMMIT" <<'PY'
import hashlib
import json
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
commit = sys.argv[2]
payload = path.read_bytes()
value = json.loads(payload.decode("ascii"))
claimed = value.pop("release_payload_sha256")
canonical = json.dumps(
    value,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
).encode("ascii")
if (
    hashlib.sha256(canonical).hexdigest() != claimed
    or value.get("status") != "pass"
    or value.get("source_commit") != commit
    or re.fullmatch(
        r"[0-9a-f]{64}",
        str(value.get("release_builder", {}).get("sha256", "")),
    )
    is None
):
    raise SystemExit("[ettr-v3-release] final release receipt differs")
print(
    json.dumps(
        {
            "release_file_sha256": hashlib.sha256(payload).hexdigest(),
            "release_payload_sha256": claimed,
            "source_commit": commit,
            "status": "pass",
        },
        sort_keys=True,
    )
)
PY
