#!/bin/bash
#SBATCH -J nmc1-eval
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/nmc1_eval_%j.out
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
ARM=${ARM:?set program or direct}
CONTROL=${CONTROL:?set normal or source_shuffled}
CHECKPOINT=${CHECKPOINT:?set immutable fitted checkpoint}
EXPECTED_CHECKPOINT_SHA256=${EXPECTED_CHECKPOINT_SHA256:?set checkpoint SHA-256}
OUT=${OUT:?set isolated output report}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL=$BASE/artifacts/external/qwen3.5-0.8b-2fc0636
DATA_ROOT=$BASE/artifacts/product_reasoning/nmc1/data_bd30f2a_r1
LAM=$BASE/artifacts/product_reasoning/lam1/full_6b1c1f3_r1/microcode.pt

case "$ARM:$CONTROL" in
  program:normal|program:source_shuffled)
    EXPECTED_TRAIN_SHA256=71010bac117d9fbca2a28e7f9a63f24456b519a7b4bc172ddd772ff27e851512
    ;;
  direct:normal)
    EXPECTED_TRAIN_SHA256=b78fa53081114f1b5c08f34c0f26df641dfd975a4975130a6b0c8098a0944757
    ;;
  *) echo "unsupported NMC1 evaluation arm/control: $ARM/$CONTROL" >&2; exit 2 ;;
esac

test ! -e "$OUT" || { echo "refusing existing output: $OUT" >&2; exit 2; }
test "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$EXPECTED_CHECKPOINT_SHA256"
test "$(sha256sum "$DATA_ROOT/development.jsonl" | awk '{print $1}')" = \
  981b83016d9a895af3016b3629c5852b868c80e364dbe97e3901a8d3e8ced4bd
test "$(sha256sum "$LAM" | awk '{print $1}')" = \
  baab62ec4a256e89950a257616d9f023b8f5e5c8c98b927052f7e722e815c7c4
test "$(sha256sum "$MODEL/tokenizer.json" | awk '{print $1}')" = \
  5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42

export PYTHONPATH="$CODE_ROOT/train:$BASE/envs/idr08-eval-r1/lib/python3.11/site-packages"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -L
"$PYTHON" "$CODE_ROOT/train/eval_nmc1_development.py" \
  --arm "$ARM" \
  --control "$CONTROL" \
  --model-root "$MODEL" \
  --model-revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --checkpoint "$CHECKPOINT" \
  --expected-train-sha256 "$EXPECTED_TRAIN_SHA256" \
  --lam-checkpoint "$LAM" \
  --data "$DATA_ROOT/development.jsonl" \
  --expected-data-sha256 981b83016d9a895af3016b3629c5852b868c80e364dbe97e3901a8d3e8ced4bd \
  --max-new-tokens 512 \
  --batch-size 8 \
  --seed 2026081053 \
  --output "$OUT"
chmod a-w "$OUT"
