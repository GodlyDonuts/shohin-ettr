# Shohin preprint build

From the repository root:

```bash
SOURCE_DATE_EPOCH=1787011200 \
  tectonic --outdir paper paper/shohin_temporal_revision.tex
```

The paper consumes the deterministic vector PDFs in
`docs/research/figures/`. Regenerate them first with:

```bash
python3 pipeline/render_shohin_publication_figure.py \
  --output-dir docs/research/figures
```
