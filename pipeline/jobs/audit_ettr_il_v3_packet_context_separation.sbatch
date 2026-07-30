#!/bin/bash
# Scan every main/reserve core by the exact terminal-packet/query-prefix
# statistic used by optimizer admission. A detected overlap is a completed
# audit with status=fail, not a scheduler/runtime failure.
#SBATCH -J shohin-ettr-packet-separation
#SBATCH -p normal
#SBATCH -c 32
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/ettr_packet_separation_%j.out
#SBATCH --open-mode=append
set -euo pipefail
export PATH=/usr/bin:/bin
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

BASE=${BASE:-/lustre/fs1/home/sa305415/shohin}
PY=${PY:-$BASE/miniforge3/bin/python}
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-32}}
SOURCE_ROOT=${SOURCE_ROOT:?set SOURCE_ROOT to a clean exact-source checkout}
SOURCE_COMMIT=${SOURCE_COMMIT:?set SOURCE_COMMIT to the exact 40-hex commit}
MAIN_AUDIT=${MAIN_AUDIT:?set MAIN_AUDIT to the passing main audit}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to immutable main materialized data}
TOKENIZER=${TOKENIZER:?set TOKENIZER to the exact corpus tokenizer}
OUT=${OUT:?set OUT to a fresh packet-separation report path}

[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  echo "[ettr-packet-separation] malformed source commit" >&2
  exit 2
}
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || {
  echo "[ettr-packet-separation] malformed worker count" >&2
  exit 2
}
(( WORKERS <= SLURM_CPUS_PER_TASK )) || {
  echo "[ettr-packet-separation] worker count exceeds allocated CPUs" >&2
  exit 2
}
[[ -x "$PY" && -d "$SOURCE_ROOT/.git" && ! -L "$SOURCE_ROOT" ]] || {
  echo "[ettr-packet-separation] source checkout or Python is unavailable" >&2
  exit 3
}
[[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || {
  echo "[ettr-packet-separation] source HEAD differs" >&2
  exit 3
}
[[ -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" ]] || {
  echo "[ettr-packet-separation] source checkout is not clean" >&2
  exit 3
}
for path in "$MAIN_AUDIT" "$TOKENIZER"; do
  [[ -f "$path" && ! -L "$path" ]] || {
    echo "[ettr-packet-separation] required file is unavailable: $path" >&2
    exit 4
  }
done
[[ -d "$DATA_ROOT" && ! -L "$DATA_ROOT" && ! -e "$OUT" ]] || {
  echo "[ettr-packet-separation] data root or fresh output differs" >&2
  exit 4
}

cd "$SOURCE_ROOT"
"$PY" - <<'PY'
import tokenizers
assert tokenizers.__version__
PY
set +e
"$PY" -u pipeline/audit_ettr_il_v3_packet_context_separation.py \
  --main-audit "$MAIN_AUDIT" \
  --data-root "$DATA_ROOT" \
  --tokenizer "$TOKENIZER" \
  --source-commit "$SOURCE_COMMIT" \
  --output "$OUT" \
  --workers "$WORKERS"
audit_rc=$?
set -e
if (( audit_rc != 0 && audit_rc != 4 )); then
  echo "[ettr-packet-separation] audit machinery failed: $audit_rc" >&2
  exit "$audit_rc"
fi
[[ -f "$OUT" && ! -L "$OUT" ]] || {
  echo "[ettr-packet-separation] report is unavailable" >&2
  exit 5
}
"$PY" - "$OUT" "$SOURCE_COMMIT" "$audit_rc" <<'PY'
import hashlib
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
commit = sys.argv[2]
audit_rc = int(sys.argv[3])
payload = path.read_bytes()
value = json.loads(payload.decode("ascii"))
claimed = value.pop("report_sha256")
canonical = (
    json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    + "\n"
).encode("ascii")
status = value.get("status")
if (
    hashlib.sha256(canonical).hexdigest() != claimed
    or value.get("source_commit") != commit
    or status not in {"pass", "fail"}
    or (audit_rc == 0) != (status == "pass")
):
    raise SystemExit("[ettr-packet-separation] final report differs")
print(
    json.dumps(
        {
            "cross_owner_components": value["cross_owner_component_count"],
            "cross_owner_contexts": value["cross_owner_context_count"],
            "report_file_sha256": hashlib.sha256(payload).hexdigest(),
            "report_sha256": claimed,
            "status": status,
            "target_conflicts": value["target_conflict_context_count"],
        },
        sort_keys=True,
    )
)
PY
