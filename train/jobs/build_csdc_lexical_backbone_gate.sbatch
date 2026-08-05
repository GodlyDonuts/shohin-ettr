#!/bin/bash
#SBATCH -J csdc-lex-data
#SBATCH -p normal
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /lustre/fs1/home/sa305415/shohin/logs/csdc_lexical_data_%j.out
#SBATCH --open-mode=append
set -euo pipefail

BASE=/lustre/fs1/home/sa305415/shohin
CODE_ROOT=${CODE_ROOT:?set immutable CODE_ROOT}
PYTHON=$BASE/miniforge3/bin/python3
TOKENIZER_SHOHIN=$BASE/artifacts/shohin-tok-32k.json
TOKENIZER_SMOL=${TOKENIZER_SMOL:?set exact SmolLM2 tokenizer path}
OUT_ROOT=${OUT_ROOT:-$BASE/artifacts/reasoning/csdc_lexical_backbone_gate_v1}
SOURCE=$OUT_ROOT/source_shohin
SMOL=$OUT_ROOT/smollm2
EXPECTED_SMOL_TOKENIZER_SHA256=9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c

export PYTHONPATH="$CODE_ROOT/train:$CODE_ROOT/pipeline"
export OMP_NUM_THREADS=8
cd "$BASE"
for path in "$TOKENIZER_SHOHIN" "$TOKENIZER_SMOL"; do
  test -f "$path" || { echo "missing input: $path" >&2; exit 2; }
done
echo "$EXPECTED_SMOL_TOKENIZER_SHA256  $TOKENIZER_SMOL" | sha256sum -c -
test ! -e "$SOURCE" || { echo "refusing existing source board: $SOURCE" >&2; exit 2; }
test ! -e "$SMOL" || { echo "refusing existing Smol board: $SMOL" >&2; exit 2; }

"$PYTHON" "$CODE_ROOT/pipeline/build_referential_literal_pointer_factorized_corpus.py" \
  --tokenizer "$TOKENIZER_SHOHIN" --out-dir "$SOURCE" \
  --train-groups 24000 --development-compositional-groups 512 \
  --development-lexical-ood-groups 512 --confirmation-groups 2048 \
  --train-seed 2026080501 --development-compositional-seed 2026080502 \
  --development-lexical-ood-seed 2026080503 --confirmation-seed 2026080504

"$PYTHON" "$CODE_ROOT/pipeline/retokenize_referential_literal_pointer_factorized_corpus.py" \
  --source-dir "$SOURCE" --tokenizer "$TOKENIZER_SMOL" --out-dir "$SMOL"

echo "[csdc-lex-data] paired corpus complete at $OUT_ROOT"
