#!/bin/bash
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
DISPATCH=$BASE/gpu_dispatch/codex_20260807_0230
BASE_RUNTIME=$BASE/runtime_overlays/diverge_cgl1_runtime_33f6f10_r2
PYTHON=$BASE/envs/product-reasoning-b3a3603-r2/bin/python
DATA=$BASE/artifacts/reasoning/diverge_ccr1/data_ec6d2f3_r1/confirmation_board.jsonl
DATA_SHA256=299237068f436ba33a68487b5300fcd724f8c98bd8bfe6b1916a4ebc7541ebf7
TOOL=$DISPATCH/eval_diverge_cgl1_hf_ceiling_3bcbc81.py
TOOL_SHA256=c8e41002e85ad67b86c9c71b796f4f381e94c5249490ef906ccaf8d223bb5f72
SLOT=${CGL1_SLOT:-$(basename "$0" .sh)}

export PYTHONPATH="$DISPATCH:$BASE_RUNTIME/train:$BASE_RUNTIME/pipeline"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/pycache
mkdir -p "$PYTHONPYCACHEPREFIX" "$BASE/logs"

test "$(sha256sum "$DATA" | awk '{print $1}')" = "$DATA_SHA256"
echo "[cgl1-dispatch] job=$SLURM_JOB_ID slot=$SLOT node=${SLURMD_NODENAME:-?}"
nvidia-smi -L

case "$SLOT" in
  slot_01|slot_02|slot_03|slot_04|slot_05|slot_06)
    test "$(sha256sum "$TOOL" | awk '{print $1}')" = "$TOOL_SHA256"
    case "$SLOT" in
      slot_01) MODEL_KIND=qwen; CONTROL=normal ;;
      slot_02) MODEL_KIND=qwen; CONTROL=scrub_context ;;
      slot_03) MODEL_KIND=qwen; CONTROL=swap_mentions ;;
      slot_04) MODEL_KIND=smollm3; CONTROL=normal ;;
      slot_05) MODEL_KIND=smollm3; CONTROL=scrub_context ;;
      slot_06) MODEL_KIND=smollm3; CONTROL=swap_mentions ;;
    esac
    if [[ "$MODEL_KIND" == qwen ]]; then
      MODEL=$BASE/artifacts/external/qwen3.5-0.8b-2fc0636
      NAME=Qwen/Qwen3.5-0.8B
      REVISION=2fc06364715b967f1860aea9cf38778875588b17
      BATCH=32
      test "$(sha256sum "$MODEL/config.json" | awk '{print $1}')" = b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204
      test "$(sha256sum "$MODEL/model.safetensors-00001-of-00001.safetensors" | awk '{print $1}')" = 04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
      test "$(sha256sum "$MODEL/tokenizer.json" | awk '{print $1}')" = 5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42
    else
      MODEL=$BASE/artifacts/external/smollm3-3b-a07cc9a
      NAME=HuggingFaceTB/SmolLM3-3B
      REVISION=a07cc9a04f16550a088caea529712d1d335b0ac1
      BATCH=8
      test "$(sha256sum "$MODEL/config.json" | awk '{print $1}')" = c72b1031274ff4626e434d0019e88e95a767460135db9ee492eb80652b786af1
      test "$(sha256sum "$MODEL/model-00001-of-00002.safetensors" | awk '{print $1}')" = e5eed4113a925264c33e3cdf76cd9f547c06a207ef2270914a7274a2bf685ffd
      test "$(sha256sum "$MODEL/model-00002-of-00002.safetensors" | awk '{print $1}')" = 6c3ad90646457295723e4da5ee8afac47006e20407b122df5174693dd8d68a43
      test "$(sha256sum "$MODEL/tokenizer.json" | awk '{print $1}')" = 7b6a500b662a34eb3f0374db856ba4ad7de4c81040571d78dc0d357238930005
    fi
    OUTPUT=$BASE/artifacts/reasoning/diverge_cgl1/capacity_3bcbc81_r1/${MODEL_KIND}_${CONTROL}.json
    test ! -e "$OUTPUT" || { echo "refusing existing CGL1 ceiling output" >&2; exit 2; }
    "$PYTHON" "$TOOL" \
      --model-root "$MODEL" --model-name "$NAME" --model-revision "$REVISION" \
      --data "$DATA" --data-sha256 "$DATA_SHA256" --control "$CONTROL" \
      --output "$OUTPUT" --batch-size "$BATCH"
    ;;
  slot_07|slot_08)
    (cd "$BASE_RUNTIME" && sha256sum -c SHA256SUMS >/dev/null)
    "$PYTHON" "$BASE_RUNTIME/train/test_diverge_cgl1_runtime.py"
    RESULTS=$BASE/artifacts/reasoning/diverge_cgl1/results_33f6f10_r1
    REPLAY_ROOT=$BASE/artifacts/reasoning/diverge_cgl1/independent_replays_3bcbc81_r1
    mkdir -p "$REPLAY_ROOT"
    if [[ "$SLOT" == slot_07 ]]; then
      ARM=shohin
      PARENT=$BASE/train/flagship_out/ckpt_0300000.pt
      PARENT_SHA256=211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6
      TOKENIZER=$BASE/artifacts/shohin-tok-32k.json
      TOKENIZER_SHA256=87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4
    else
      ARM=smollm2
      PARENT=$BASE/train/ettr_smollm2_control_parent_a2026072801_7881d8e/joint-model-final.pt
      PARENT_SHA256=8196f810a31e0abe7f3bf0eae0a37b103195f109b7a8e962c7b74b5710c98a02
      TOKENIZER=$BASE/artifacts/external/smollm2_135m_instruct_83212e1e/tokenizer.json
      TOKENIZER_SHA256=9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c
    fi
    CHECKPOINT=$RESULTS/$ARM/model/checkpoint.pt
    SOURCE_RESULT=$RESULTS/$ARM/development.json
    for _ in $(seq 1 600); do
      [[ -s "$CHECKPOINT" && -s "$SOURCE_RESULT" ]] && break
      sleep 5
    done
    test -s "$CHECKPOINT" && test -s "$SOURCE_RESULT"
    CHECKPOINT_SHA256=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
    # Primary arm directories become immutable before the source report exists.
    OUTPUT=$REPLAY_ROOT/${ARM}_development_${SLURM_JOB_ID}.json
    test ! -e "$OUTPUT"
    "$PYTHON" "$BASE_RUNTIME/train/eval_diverge_cgl1.py" \
      --checkpoint "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA256" \
      --base "$PARENT" --base-sha256 "$PARENT_SHA256" \
      --tokenizer "$TOKENIZER" --tokenizer-sha256 "$TOKENIZER_SHA256" \
      --data "$DATA" --data-sha256 "$DATA_SHA256" --board-type development \
      --output "$OUTPUT" --batch-size 32 --device cuda
    ;;
  *) echo "unknown CGL1 dispatch slot: $SLOT" >&2; exit 2 ;;
esac

echo "[cgl1-dispatch] completed job=$SLURM_JOB_ID slot=$SLOT"
