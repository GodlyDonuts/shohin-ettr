#!/bin/bash
#SBATCH -J cgl1-orbit-diag
#SBATCH -p normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH --exclude=evc33,evc43
#SBATCH -o /lustre/fs1/home/%u/shohin/logs/cgl1-orbit-diag-%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
BASE_CODE_ROOT=${BASE_CODE_ROOT:?set immutable CGL1 base runtime}
OVERLAY_ROOT=${OVERLAY_ROOT:?set immutable orbit-attribution overlay}
OVERLAY_SHA256SUMS_SHA256=${OVERLAY_SHA256SUMS_SHA256:?set overlay receipt hash}
ARM=${ARM:?set shohin or smollm2}
OUTPUT=${OUTPUT:?set fresh attribution report}
PYTHON=${PYTHON:-$BASE/envs/product-reasoning-b3a3603-r2/bin/python}
DATA=$BASE/artifacts/reasoning/diverge_ccr1/data_ec6d2f3_r1/confirmation_board.jsonl
DATA_SHA256=299237068f436ba33a68487b5300fcd724f8c98bd8bfe6b1916a4ebc7541ebf7

case "$ARM" in
  shohin)
    CHECKPOINT=$BASE/artifacts/reasoning/diverge_cgl1/results_33f6f10_r1/shohin/model/checkpoint.pt
    CHECKPOINT_SHA256=7b15104922ad2dbfd0c4ce48f3425e47ef6eafdb8cd5215cf97f772c2614cf3b
    SOURCE_RESULT=$BASE/artifacts/reasoning/diverge_cgl1/results_33f6f10_r1/shohin/development.json
    SOURCE_RESULT_SHA256=a02479576a5d404f3d5a7bc55d5218652877656db7c76dd8e130dd46d431a38c
    PARENT=$BASE/train/flagship_out/ckpt_0300000.pt
    PARENT_SHA256=211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6
    TOKENIZER=$BASE/artifacts/shohin-tok-32k.json
    TOKENIZER_SHA256=87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4
    ;;
  smollm2)
    CHECKPOINT=$BASE/artifacts/reasoning/diverge_cgl1/results_33f6f10_r1/smollm2/model/checkpoint.pt
    CHECKPOINT_SHA256=6deb7d2d9b3a6809c903d53212d7803c0a7b6f11319ef9716ac3c91b946176a3
    SOURCE_RESULT=$BASE/artifacts/reasoning/diverge_cgl1/results_33f6f10_r1/smollm2/development.json
    SOURCE_RESULT_SHA256=5b788fa7fe8a79e59e80b814c09f2d4084868450b64b1f458c6b83f99058eaf4
    PARENT=$BASE/train/ettr_smollm2_control_parent_a2026072801_7881d8e/joint-model-final.pt
    PARENT_SHA256=8196f810a31e0abe7f3bf0eae0a37b103195f109b7a8e962c7b74b5710c98a02
    TOKENIZER=$BASE/artifacts/external/smollm2_135m_instruct_83212e1e/tokenizer.json
    TOKENIZER_SHA256=9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c
    ;;
  *) echo "invalid CGL1 attribution ARM=$ARM" >&2; exit 2 ;;
esac

test ! -e "$OUTPUT"
test "$(sha256sum "$OVERLAY_ROOT/SHA256SUMS" | awk '{print $1}')" = "$OVERLAY_SHA256SUMS_SHA256"
(cd "$OVERLAY_ROOT" && sha256sum -c SHA256SUMS >/dev/null)
for path in "$CHECKPOINT" "$SOURCE_RESULT" "$PARENT" "$TOKENIZER" "$DATA"; do
  test -f "$path" || { echo "missing CGL1 attribution input: $path" >&2; exit 2; }
done

export PYTHONPATH="$OVERLAY_ROOT/train:$BASE_CODE_ROOT/train:$BASE_CODE_ROOT/pipeline"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$(dirname "$OUTPUT")"
echo "[cgl1-orbit] job=$SLURM_JOB_ID arm=$ARM node=${SLURMD_NODENAME:-?}"
nvidia-smi -L
"$PYTHON" "$OVERLAY_ROOT/train/test_diagnose_diverge_cgl1_orbit_quotient.py"
"$PYTHON" "$OVERLAY_ROOT/train/diagnose_diverge_cgl1_orbit_quotient.py" \
  --checkpoint "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --source-result "$SOURCE_RESULT" --source-result-sha256 "$SOURCE_RESULT_SHA256" \
  --base "$PARENT" --base-sha256 "$PARENT_SHA256" \
  --tokenizer "$TOKENIZER" --tokenizer-sha256 "$TOKENIZER_SHA256" \
  --data "$DATA" --data-sha256 "$DATA_SHA256" \
  --output "$OUTPUT" --batch-size 32 --device cuda
echo "[cgl1-orbit] completed job=$SLURM_JOB_ID arm=$ARM"
