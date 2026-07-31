#!/bin/bash
#SBATCH --job-name=shohin-joint-board
#SBATCH --partition=normal
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc32,evc33,evc40,evc43,evc44
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/joint-base-board-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/joint-base-board-%j.err

set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set immutable source root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set exact source commit}
RUNTIME_SHA256SUMS_SHA256=${RUNTIME_SHA256SUMS_SHA256:?set runtime receipt hash}
RUN_DIR=${RUN_DIR:?set completed joint run directory}
TOKENIZER=${TOKENIZER:?set tokenizer path}
EVAL_ROOT=${EVAL_ROOT:?set immutable evaluation root}
OUTPUT_DIR=${OUTPUT_DIR:?set fresh board output directory}
PYTHON_ROOT=${PYTHON_ROOT:-/lustre/fs1/home/sa305415/shohin/miniforge3}
N=${N:-200}
GSM_K=${GSM_K:-4}
MAX_NEW=${MAX_NEW:-256}
EVAL_SEED=${EVAL_SEED:-20260712}

for value in "$CODE_ROOT" "$RUN_DIR" "$TOKENIZER" "$EVAL_ROOT" \
  "$OUTPUT_DIR" "$PYTHON_ROOT"; do
  [[ "$value" == /* ]] || {
    echo "joint board paths must be absolute" >&2
    exit 2
  }
done
if [[ -e "$OUTPUT_DIR" || -L "$OUTPUT_DIR" ]]; then
  echo "joint board output exists" >&2
  exit 2
fi
cd "$CODE_ROOT"
[[ "$(tr -d '\r\n' < SOURCE_COMMIT)" == "$SOURCE_COMMIT" ]] \
  && [[ "$(sha256sum SHA256SUMS | awk '{print $1}')" \
    == "$RUNTIME_SHA256SUMS_SHA256" ]] \
  && [[ -z "$(sha256sum -c SHA256SUMS 2>&1 | grep -v ': OK$')" ]] \
  || {
    echo "joint board runtime differs" >&2
    exit 2
  }
(
  cd "$RUN_DIR"
  sha256sum -c SHA256SUMS
)
checkpoint="$RUN_DIR/base-eval-final.pt"
[[ -s "$checkpoint" ]] || {
  echo "joint base checkpoint is missing" >&2
  exit 2
}
mkdir -m 700 "$OUTPUT_DIR"
log="$OUTPUT_DIR/board.log"
summary="$OUTPUT_DIR/summary.jsonl"
touch "$log" "$summary"
chmod 600 "$log" "$summary"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2

if ! "$PYTHON_ROOT/bin/python" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit(75)
if "H100" not in torch.cuda.get_device_name(0).upper():
    sys.exit(76)
torch.ones(1, device="cuda").add_(1)
torch.cuda.synchronize()
PY
then
  echo "joint board CUDA preflight failed" >&2
  exit 75
fi

run() {
  "$@" 2>&1 | tee -a "$log"
}
run "$PYTHON_ROOT/bin/python" "$CODE_ROOT/train/eval_suite.py" \
  --ckpt "$checkpoint" --tokenizer "$TOKENIZER" \
  --task gsm8k --data "$EVAL_ROOT/gsm8k.jsonl" \
  --n "$N" --k "$GSM_K" --temp 0.8 --max-new "$MAX_NEW" \
  --seed "$EVAL_SEED"
run "$PYTHON_ROOT/bin/python" "$CODE_ROOT/train/eval_suite.py" \
  --ckpt "$checkpoint" --tokenizer "$TOKENIZER" \
  --task gsm8k --data "$EVAL_ROOT/gsm8k.jsonl" \
  --n "$N" --k 1 --max-new "$MAX_NEW" --seed "$EVAL_SEED"
run "$PYTHON_ROOT/bin/python" "$CODE_ROOT/train/eval_suite.py" \
  --ckpt "$checkpoint" --tokenizer "$TOKENIZER" \
  --task math500 --data "$EVAL_ROOT/math500.jsonl" \
  --n "$N" --k 1 --max-new "$MAX_NEW" --seed "$EVAL_SEED"
run "$PYTHON_ROOT/bin/python" "$CODE_ROOT/train/eval_code.py" \
  --ckpt "$checkpoint" --tokenizer "$TOKENIZER" \
  --task humaneval --data "$EVAL_ROOT/humaneval_full.jsonl" \
  --n 164 --k 1 --seed "$EVAL_SEED"
run "$PYTHON_ROOT/bin/python" "$CODE_ROOT/train/eval_code.py" \
  --ckpt "$checkpoint" --tokenizer "$TOKENIZER" \
  --task mbpp --data "$EVAL_ROOT/mbpp_full.jsonl" \
  --n "$N" --k 1 --seed "$EVAL_SEED"

"$PYTHON_ROOT/bin/python" - "$log" "$summary" <<'PY'
import json
import re
import sys

log_path, summary_path = sys.argv[1:]
pattern = re.compile(
    r"^(gsm8k|math500|humaneval|mbpp)\s+"
    r"(maj@\d+|pass@\d+)\s+"
    r"ckpt=(.*?)\s+step=(.*?)\s+"
    r"(\d+)/(\d+)\s+=\s+([0-9.]+)%"
)
rows = []
with open(log_path, errors="replace") as source:
    for line in source:
        match = pattern.search(line.strip())
        if match:
            rows.append(
                {
                    "accuracy": int(match.group(5))
                    / max(1, int(match.group(6))),
                    "checkpoint": match.group(3),
                    "correct": int(match.group(5)),
                    "metric": match.group(2),
                    "percent": float(match.group(7)),
                    "step": match.group(4),
                    "task": match.group(1),
                    "total": int(match.group(6)),
                }
            )
if len(rows) != 5:
    raise SystemExit("joint board did not produce five metric rows")
with open(summary_path, "w") as output:
    for row in rows:
        output.write(json.dumps(row, sort_keys=True) + "\n")
PY
(
  cd "$OUTPUT_DIR"
  sha256sum board.log summary.jsonl > SHA256SUMS
  chmod 400 board.log summary.jsonl SHA256SUMS
  sha256sum -c SHA256SUMS
)
