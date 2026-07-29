#!/bin/bash
#SBATCH -J shohin-fw-batch
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/fineweb_edu_batch_canary_%j.out
#SBATCH --open-mode=append
# Compare tokenizer batching only. This job never writes a training candidate.
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
PY=$BASE/miniforge3/bin/python
CODE_ROOT=${CODE_ROOT:?set CODE_ROOT to an isolated, pinned worktree}
EXPECTED_TOKENIZE_SHA256=${EXPECTED_TOKENIZE_SHA256:?set EXPECTED_TOKENIZE_SHA256}
TOKENIZE=$CODE_ROOT/pipeline/tokenize_shards.py
VERIFY=$CODE_ROOT/pipeline/verify_tokenized_shards.py
INPUT_FILE=${INPUT_FILE:-$BASE/scratchpad/fineweb_edu_sample100bt_87f0914/000_00000.parquet}
MAX_TOKENS=${MAX_TOKENS:-2000000}
CANARY_ROOT=${CANARY_ROOT:-$BASE/artifacts/shards/fineweb_edu_tokenizer_batch_canary_${SLURM_JOB_ID}}
OUT_ONE=$CANARY_ROOT/batch_1
OUT_MANY=$CANARY_ROOT/batch_256

[[ -d "$CODE_ROOT" && ! -L "$CODE_ROOT" ]] || {
  echo "[fineweb-batch] isolated code root is unavailable" >&2
  exit 2
}
[[ -f "$TOKENIZE" && ! -L "$TOKENIZE" ]] || {
  echo "[fineweb-batch] tokenizer source is unavailable" >&2
  exit 2
}
printf '%s  %s\n' "$EXPECTED_TOKENIZE_SHA256" "$TOKENIZE" | sha256sum -c -
[[ -f "$INPUT_FILE" && ! -L "$INPUT_FILE" ]] || {
  echo "[fineweb-batch] input must be a physical parquet file" >&2
  exit 3
}
[[ ! -e "$CANARY_ROOT" ]] || {
  echo "[fineweb-batch] canary root already exists: $CANARY_ROOT" >&2
  exit 4
}
[[ "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] || {
  echo "[fineweb-batch] MAX_TOKENS must be positive" >&2
  exit 5
}

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
export RAYON_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true
mkdir -p "$CANARY_ROOT"

run_one() {
  local batch_size=$1
  local output_dir=$2
  local started elapsed
  started=$(date +%s)
  "$PY" -u "$TOKENIZE" \
    --tokenizer "$BASE/artifacts/shohin-tok-32k.json" \
    --dataset HuggingFaceFW/fineweb-edu \
    --config sample-100BT \
    --revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 \
    --input-files "$INPUT_FILE" \
    --text-col text \
    --lang en \
    --lang-field language \
    --require-lang-field \
    --min-number-field int_score \
    --min-number 4 \
    --decontam-grams "$BASE/artifacts/evals/evalgrams.pkl" \
    --eval-glob "$BASE"/artifacts/evals/*.jsonl \
    --min-chars 400 \
    --max-chars 100000 \
    --exact-dedup \
    --max-line-repeat-fraction 0.30 \
    --max-boilerplate-markers 1 \
    --min-alpha-fraction 0.55 \
    --max-control-fraction 0.001 \
    --max-replacement-fraction 0.0005 \
    --domain-field url \
    --max-tokens-per-domain 5000000 \
    --out-dir "$output_dir" \
    --shard-tokens 100000000 \
    --max-tokens "$MAX_TOKENS" \
    --tokenizer-batch-size "$batch_size"
  RUN_SECONDS=$(( $(date +%s) - started ))
  echo "[fineweb-batch] batch=$batch_size elapsed_seconds=$RUN_SECONDS"
}

run_one 1 "$OUT_ONE"
SECONDS_ONE=$RUN_SECONDS
PYTHONPATH="$CODE_ROOT" "$PY" -m pipeline.verify_tokenized_shards \
  --shard-dir "$OUT_ONE" \
  --selection-code "$TOKENIZE" \
  --require-external-inputs

run_one 256 "$OUT_MANY"
SECONDS_MANY=$RUN_SECONDS
PYTHONPATH="$CODE_ROOT" "$PY" -m pipeline.verify_tokenized_shards \
  --shard-dir "$OUT_MANY" \
  --selection-code "$TOKENIZE" \
  --require-external-inputs

"$PY" - "$OUT_ONE" "$OUT_MANY" "$CANARY_ROOT/report.json" \
  "$SECONDS_ONE" "$SECONDS_MANY" "$MAX_TOKENS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

one = Path(sys.argv[1])
many = Path(sys.argv[2])
report_path = Path(sys.argv[3])
seconds_one = int(sys.argv[4])
seconds_many = int(sys.argv[5])
max_tokens = int(sys.argv[6])


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalized_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    manifest["tokenizer"]["batch_size"] = "normalized-for-parity"
    manifest["payload_sha256"] = "normalized-for-parity"
    return manifest


one_files = sorted(
    path.relative_to(one).as_posix() for path in one.rglob("*") if path.is_file()
)
many_files = sorted(
    path.relative_to(many).as_posix() for path in many.rglob("*") if path.is_file()
)
if one_files != many_files:
    raise SystemExit("batch canary file sets differ")
for relative in one_files:
    if relative == "manifest.json":
        continue
    if file_digest(one / relative) != file_digest(many / relative):
        raise SystemExit(f"batch canary artifact differs: {relative}")

manifest_one = normalized_manifest(one / "manifest.json")
manifest_many = normalized_manifest(many / "manifest.json")
if manifest_one != manifest_many:
    raise SystemExit("batch canary normalized manifests differ")
tokens = manifest_one["tokens"]
if tokens <= 0:
    raise SystemExit("batch canary retained no tokens")
report = {
    "schema": "shohin-fineweb-tokenizer-batch-canary-v1",
    "max_tokens": max_tokens,
    "artifacts_exact": True,
    "tokens": tokens,
    "batch_1_seconds": seconds_one,
    "batch_256_seconds": seconds_many,
    "batch_1_tokens_per_second": tokens / seconds_one,
    "batch_256_tokens_per_second": tokens / seconds_many,
    "speedup": seconds_one / seconds_many,
    "manifest_payload_sha256": json.loads((one / "manifest.json").read_text())["payload_sha256"],
}
report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, sort_keys=True))
PY

find "$CANARY_ROOT" -type f -exec chmod 0444 {} +
find "$CANARY_ROOT" -type d -exec chmod 0555 {} +
echo "[fineweb-batch] immutable audit canary committed at $CANARY_ROOT"
