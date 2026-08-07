#!/bin/bash
#SBATCH -J compose-qte1
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 00:45:00
#SBATCH --exclude=evc33,evc43
#SBATCH -o /lustre/fs1/home/%u/shohin/logs/compose_qte1_%j.out
#SBATCH --open-mode=append
set -euo pipefail

ROOT=/lustre/fs1/home/sa305415/shohin
BASE_CODE_ROOT=${BASE_CODE_ROOT:?set immutable qualified base code root}
OVERLAY_ROOT=${OVERLAY_ROOT:?set immutable QTE1 composite overlay}
SOURCE_COMMIT=${SOURCE_COMMIT:?set QTE1 composite commit}
DIRECT_CONFIRMATION=${DIRECT_CONFIRMATION:?set passing QTE1 confirmation}
OUTPUT=${OUTPUT:?set isolated QTE1 composite output}
PYTHON=${PYTHON:-$ROOT/envs/product-reasoning-b3a3603-r2/bin/python}
MODEL=$ROOT/artifacts/external/qwen3.5-0.8b-2fc0636
STAGE_CHECKPOINT=$ROOT/artifacts/reasoning/diverge_rrg1/model_f23d032_r1/checkpoint_0002000.pt
STAGE_CHECKPOINT_SHA256=05c7b6c7ee68e14133231d572b8c66aced5e3a72a6ba25827498560f723336ae
DATA=$ROOT/artifacts/reasoning/diverge_pqi1/data_defc87f_r1/confirmation_board.jsonl
DATA_SHA256=27f198680cc7bcd7e0203949fe4dee1658fc057fe75302b7e2d43d74321201b8
PROTECTED_NVE1=$ROOT/artifacts/reasoning/diverge_nve1/result_fbb2f33_r1/evaluation.json
PROTECTED_NVE1_SHA256=2cda05808f0e5230fd285c345a7431b6791a22b3c35c62b227df35c0f622d6c3
PROTECTED_TOL3=$ROOT/artifacts/reasoning/diverge_tol3_0dfaf53_r1/confirmation_evaluation_0dfaf53_r1.json
PROTECTED_TOL3_SHA256=2f5b1ca3c08f4b82f8e220713441014178da72a257b8fafb67b0887f28ad5700

export PYTHONPATH="$OVERLAY_ROOT/train:$BASE_CODE_ROOT/train:$BASE_CODE_ROOT/pipeline"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$ROOT/logs"
cd "$BASE_CODE_ROOT"

check_hash() {
  local path=$1
  local expected=$2
  test -f "$path" || { echo "missing QTE1 composite input: $path" >&2; exit 2; }
  test "$(sha256sum "$path" | awk '{print $1}')" = "$expected" || {
    echo "QTE1 composite input hash differs: $path" >&2
    exit 2
  }
}

check_hash "$STAGE_CHECKPOINT" "$STAGE_CHECKPOINT_SHA256"
check_hash "$DATA" "$DATA_SHA256"
check_hash "$PROTECTED_NVE1" "$PROTECTED_NVE1_SHA256"
check_hash "$PROTECTED_TOL3" "$PROTECTED_TOL3_SHA256"
check_hash "$MODEL/config.json" b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204
check_hash "$MODEL/model.safetensors-00001-of-00001.safetensors" 04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
check_hash "$MODEL/tokenizer.json" 5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42
test -f "$DIRECT_CONFIRMATION" || { echo "missing QTE1 direct confirmation" >&2; exit 2; }
test ! -e "$OUTPUT" || { echo "refusing existing QTE1 composite" >&2; exit 2; }

echo "[compose-qte1] node=${SLURMD_NODENAME:-?} commit=$SOURCE_COMMIT"
nvidia-smi -L
"$PYTHON" "$OVERLAY_ROOT/train/test_diverge_qte1_composite.py"
"$PYTHON" "$OVERLAY_ROOT/train/eval_diverge_qte1_composite.py" \
  --source-commit "$SOURCE_COMMIT" \
  --stage-checkpoint "$STAGE_CHECKPOINT" \
  --stage-checkpoint-sha256 "$STAGE_CHECKPOINT_SHA256" \
  --model-root "$MODEL" \
  --direct-confirmation "$DIRECT_CONFIRMATION" \
  --data "$DATA" --data-sha256 "$DATA_SHA256" \
  --protected-nve1-result "$PROTECTED_NVE1" \
  --protected-nve1-result-sha256 "$PROTECTED_NVE1_SHA256" \
  --protected-tol3-result "$PROTECTED_TOL3" \
  --protected-tol3-result-sha256 "$PROTECTED_TOL3_SHA256" \
  --output "$OUTPUT" --batch-size 32
