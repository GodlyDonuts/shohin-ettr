#!/bin/bash
#SBATCH --job-name=shohin-ettr-tri
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc32,evc33,evc40,evc43,evc44
#SBATCH --output=/lustre/fs1/home/sa305415/shohin/logs/joint-ettr-tri-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/shohin/logs/joint-ettr-tri-%j.err

set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set immutable source root}
SOURCE_COMMIT=${SOURCE_COMMIT:?set exact source commit}
RUNTIME_SHA256SUMS_SHA256=${RUNTIME_SHA256SUMS_SHA256:?set runtime receipt hash}
RELEASE_ROOT=${RELEASE_ROOT:?set ETTR release root}
RELEASE_SHA256=${RELEASE_SHA256:?set ETTR release hash}
ETTR_DATA_ROOT=${ETTR_DATA_ROOT:?set ETTR source root}
TOKENIZER=${TOKENIZER:?set tokenizer path}
GENERAL_SHARD_ROOT=${GENERAL_SHARD_ROOT:?set historical shard root}
PARENT_RUN_DIR=${PARENT_RUN_DIR:?set completed parent joint run}
INSTRUCTION_DATA=${INSTRUCTION_DATA:?set admitted instruction JSONL}
INSTRUCTION_DATA_SHA256=${INSTRUCTION_DATA_SHA256:?set instruction hash}
INSTRUCTION_QUALITY=${INSTRUCTION_QUALITY:?set quality report}
INSTRUCTION_FULL_TEXT=${INSTRUCTION_FULL_TEXT:?set full-text report}
OUTDIR=${OUTDIR:?set fresh output directory}
DATA_SEED=${DATA_SEED:?set data seed}
PYTHON_ROOT=${PYTHON_ROOT:-/lustre/fs1/home/sa305415/shohin/miniforge3}
UPDATES=${UPDATES:-6000}
GENERAL_POSITION_WEIGHT=${GENERAL_POSITION_WEIGHT:-15}
INSTRUCTION_POSITION_WEIGHT=${INSTRUCTION_POSITION_WEIGHT:-70}
ETTR_POSITION_WEIGHT=${ETTR_POSITION_WEIGHT:-15}
QUERY_BINDING_WEIGHT=${QUERY_BINDING_WEIGHT:-1}
QUERY_BINDING_REDUCTION=${QUERY_BINDING_REDUCTION:-mixed-mean}
QUERY_BINDING_CLASSIFICATION_WEIGHT=${QUERY_BINDING_CLASSIFICATION_WEIGHT:-1}
QUERY_BINDING_EFFECT_WEIGHT=${QUERY_BINDING_EFFECT_WEIGHT:-1}
QUERY_BINDING_INVARIANCE_WEIGHT=${QUERY_BINDING_INVARIANCE_WEIGHT:-1}
QUERY_BINDING_RISK_TEMPERATURE=${QUERY_BINDING_RISK_TEMPERATURE:-1}
PACKET_WEIGHT=${PACKET_WEIGHT:-1}
INTERVENTION_WEIGHT=${INTERVENTION_WEIGHT:-1}
TRANSACTION_WEIGHT=${TRANSACTION_WEIGHT:-1}
COMMIT_HALT_WEIGHT=${COMMIT_HALT_WEIGHT:-0.5}
GRADIENT_CLIP_MODE=${GRADIENT_CLIP_MODE:-owner}

for path in \
  "$CODE_ROOT" "$RELEASE_ROOT" "$ETTR_DATA_ROOT" "$TOKENIZER" \
  "$GENERAL_SHARD_ROOT" "$PARENT_RUN_DIR" "$INSTRUCTION_DATA" \
  "$INSTRUCTION_QUALITY" "$INSTRUCTION_FULL_TEXT" "$OUTDIR" \
  "$PYTHON_ROOT"; do
  [[ "$path" == /* ]] || {
    echo "tri-stream paths must be absolute" >&2
    exit 2
  }
done
if [[ ! "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ \
  || ! "$RUNTIME_SHA256SUMS_SHA256" =~ ^[0-9a-f]{64}$ \
  || ! "$RELEASE_SHA256" =~ ^[0-9a-f]{64}$ \
  || ! "$INSTRUCTION_DATA_SHA256" =~ ^[0-9a-f]{64}$ \
  || ! "$DATA_SEED" =~ ^[0-9]+$ \
  || ! "$UPDATES" =~ ^[1-9][0-9]*$ \
  || ! "$GENERAL_POSITION_WEIGHT" =~ ^[1-9][0-9]*$ \
  || ! "$INSTRUCTION_POSITION_WEIGHT" =~ ^[1-9][0-9]*$ \
  || ! "$ETTR_POSITION_WEIGHT" =~ ^[1-9][0-9]*$ ]]; then
  echo "tri-stream launch identity differs" >&2
  exit 2
fi
if [[ -e "$OUTDIR" || -L "$OUTDIR" ]]; then
  echo "tri-stream output already exists" >&2
  exit 2
fi

cd "$CODE_ROOT"
[[ "$(tr -d '\r\n' < SOURCE_COMMIT)" == "$SOURCE_COMMIT" ]] \
  && [[ "$(sha256sum SHA256SUMS | awk '{print $1}')" \
    == "$RUNTIME_SHA256SUMS_SHA256" ]] \
  && [[ -z "$(sha256sum -c SHA256SUMS 2>&1 | grep -v ': OK$')" ]] \
  || {
    echo "tri-stream runtime differs" >&2
    exit 2
  }
(
  cd "$PARENT_RUN_DIR"
  sha256sum -c SHA256SUMS
)
parent_model="$PARENT_RUN_DIR/joint-model-final.pt"
parent_sha=$(sha256sum "$parent_model" | awk '{print $1}')
[[ "$(sha256sum "$INSTRUCTION_DATA" | awk '{print $1}')" \
  == "$INSTRUCTION_DATA_SHA256" ]] || {
  echo "tri-stream instruction data differs" >&2
  exit 2
}
"$PYTHON_ROOT/bin/python" - \
  "$INSTRUCTION_DATA" "$INSTRUCTION_DATA_SHA256" \
  "$INSTRUCTION_QUALITY" "$INSTRUCTION_FULL_TEXT" <<'PY'
import json
import sys

data, digest, quality_path, text_path = sys.argv[1:]
quality = json.load(open(quality_path))
text = json.load(open(text_path))
if quality.get("data_sha256") != digest:
    raise SystemExit("instruction quality hash differs")
if (
    quality.get("malformed_rows")
    or quality.get("missing_question_or_response")
    or quality.get("duplicate_normalized_questions")
):
    raise SystemExit("instruction quality gate failed")
overlap = quality.get("eval_overlap", {})
if overlap.get("exact_prompt_hits") or overlap.get("ngram_prompt_hits"):
    raise SystemExit("instruction prompt overlap gate failed")
if text.get("data_sha256") != digest:
    raise SystemExit("instruction full-text hash differs")
full_overlap = text.get("overlap", {})
if full_overlap.get("exact_rows") or full_overlap.get("ngram_rows"):
    raise SystemExit("instruction full-text overlap gate failed")
PY
for name in finemath4 openwebmath code_python finemath3; do
  [[ -d "$GENERAL_SHARD_ROOT/$name" ]] || {
    echo "legacy general shard directory is missing" >&2
    exit 2
  }
done

"$PYTHON_ROOT/bin/python" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit(75)
if "H100" not in torch.cuda.get_device_name(0).upper():
    sys.exit(76)
torch.ones(1, device="cuda").add_(1)
torch.cuda.synchronize()
PY

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2
args=(
  --release-root "$RELEASE_ROOT"
  --release-sha256 "$RELEASE_SHA256"
  --ettr-data-root "$ETTR_DATA_ROOT"
  --tokenizer "$TOKENIZER"
  --parent-joint-model "$parent_model"
  --parent-joint-model-sha256 "$parent_sha"
  --instruction-data "$INSTRUCTION_DATA"
  --instruction-data-sha256 "$INSTRUCTION_DATA_SHA256"
  --instruction-sample-weight math=0.40
  --instruction-sample-weight procedural=0.30
  --instruction-sample-weight code=0.20
  --instruction-sample-weight teacher=0.05
  --output "$OUTDIR"
  --source-commit "$SOURCE_COMMIT"
  --updates "$UPDATES"
  --total-updates "$UPDATES"
  --warmup-updates 200
  --general-batch-size 16
  --instruction-batch-size 16
  --general-position-weight "$GENERAL_POSITION_WEIGHT"
  --instruction-position-weight "$INSTRUCTION_POSITION_WEIGHT"
  --ettr-position-weight "$ETTR_POSITION_WEIGHT"
  --data-seed "$DATA_SEED"
  --base-lr-muon 0.0015
  --base-lr-adam 0.00035
  --architecture-lr-muon 0.003
  --architecture-lr-adam 0.0006
  --nll-gradient-cap 4
  --query-binding-weight "$QUERY_BINDING_WEIGHT"
  --query-binding-reduction "$QUERY_BINDING_REDUCTION"
  --query-binding-classification-weight \
    "$QUERY_BINDING_CLASSIFICATION_WEIGHT"
  --query-binding-effect-weight "$QUERY_BINDING_EFFECT_WEIGHT"
  --query-binding-invariance-weight "$QUERY_BINDING_INVARIANCE_WEIGHT"
  --query-binding-risk-temperature "$QUERY_BINDING_RISK_TEMPERATURE"
  --packet-weight "$PACKET_WEIGHT"
  --intervention-weight "$INTERVENTION_WEIGHT"
  --transaction-weight "$TRANSACTION_WEIGHT"
  --commit-halt-weight "$COMMIT_HALT_WEIGHT"
  --gradient-clip-mode "$GRADIENT_CLIP_MODE"
)
for name in finemath4 openwebmath code_python finemath3; do
  args+=(--legacy-general-shard-dir "$GENERAL_SHARD_ROOT/$name")
  args+=(--legacy-general-weight 1)
done
"$PYTHON_ROOT/bin/python" \
  "$CODE_ROOT/train/train_ettr_joint_instruction_canary.py" \
  "${args[@]}"

(
  cd "$OUTDIR"
  sha256sum \
    base-eval-final.pt \
    final-report.json \
    joint-model-final.pt \
    run-contract.json \
    train.jsonl \
    > SHA256SUMS
  chmod 400 SHA256SUMS
  sha256sum -c SHA256SUMS
)
