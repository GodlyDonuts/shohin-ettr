# Q36 causal-transfer synthesis

Status: prepared post-terminal analysis; non-gating; no scientific job authorized.

This analysis connects the frozen Q36 MoE architecture result to the already measured DSET/ISET causal boundary without changing either experiment. It answers a narrower question than the Q36 gate: if Q36 reaches a formal terminal result, does its paired `trained revision vs draft-hidden` contrast reproduce the draft-information mechanism previously observed under controlled edit interventions?

The reducer is [`pipeline/synthesize_q36_causal_transfer.py`](../../pipeline/synthesize_q36_causal_transfer.py). It accepts only:

- a complete `shohin-q36-mtr-final-comparison-v1` terminal result;
- the byte-exact closed DSET1 result (`7915c924…c44a`);
- the byte-exact closed ISET1 result (`b62897ac…5f5d`); and
- the byte-exact trained Q35 transfer result (`c0452f9a…bfe9`).

It preserves three logically separate conclusions:

1. The formal Q36 PASS/FAIL remains exactly the formal architecture result.
2. Draft-information mechanism transfer requires Q36's Holm-corrected paired revision-over-draft-hidden claim plus positive aligned-over-hidden effects in DSET1, ISET1, and trained Q35 transfer.
3. The action-selection bottleneck is supported only at the prior measured boundary: natural choice script accuracy was 177/256, while externally fixing the action yielded 128/128 script and execution correctness on all faulted choice rows.

No cross-board absolute scores are compared, no observations are statistically pooled across boards, and no scaling-law claim is authorized. A Q36 FAIL cannot become an architecture success through this analysis. Conversely, a supported causal mechanism may be reported alongside a Q36 FAIL, but only as a non-gating mechanistic result.

The intended terminal invocation is:

```bash
python pipeline/synthesize_q36_causal_transfer.py \
  --q36-terminal /path/to/final_comparison.json \
  --dset1 docs/research/SHOHIN_DSET1_RESULT.json \
  --iset1 docs/research/SHOHIN_ISET1_RESULT.json \
  --q35-trained docs/research/SHOHIN_DSET_Q35_TRAINED_TRANSFER_RESULT.json \
  --output /fresh/evidence/q36_causal_transfer_synthesis.json
```

The output is write-once and explicitly authorizes no retry, confirmation, successor, or new scientific job.
