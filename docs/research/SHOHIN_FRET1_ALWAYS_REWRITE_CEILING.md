# FRET1: Frozen Always-Rewrite Transduction Ceiling

**Status:** prospectively frozen before output, 2026-08-10  
**Scope:** source-disjoint development diagnostic only; holdout sealed

## Hypothesis

DSET-Q35 proves that the frozen host can emit an exact old-surface content
pointer and corrected replacement on 128/128 faulted choice rows when its
`REPLACE_LAST` action is supplied. GSET and ISET show that learned binary
fault detection, not deterministic execution, remains the bottleneck. PSET1
already tested a separate small pointer/value module and closed because its
standalone byte decoder achieved only 12.89% exact replacement.

FRET1 is a read-only ceiling, not a new fitted model. It supplies the fixed
architecture token `<REPLACE_LAST>` to the frozen aligned DSET-Q35 checkpoint
for every clean and fault presentation. The model must emit the exact old
surface and new surface. A generic deterministic executor copies the complete
draft except that model-owned content-addressed span. Clean rows require an
idempotent `old -> old` transaction; fault rows require `old -> corrected`.
The final trajectory therefore cannot be materialized without the draft.

The exact draft-hidden DSET checkpoint is evaluated with identical prompts,
prefix, decoding, sharding, and executor. No fit, verifier, answer label,
semantic host repair, public benchmark, or holdout is involved.

## Frozen execution

- pinned Qwen3.6-35B-A3B revision `995ad96e`;
- aligned DSET checkpoint SHA-256
  `166d8cbafd5fa5aa842a42c8856294436ed2704db3d51396a3554576399ea8bb`;
- exact hidden-arm checkpoint and hashes bound at launch;
- exact 1,908-row DSET-Q35 development diagnostic, 8 shards per arm;
- greedy decoding, 31 generated suffix tokens, fixed seed `2026081010`;
- no model updates and no accepted malformed or exhausted output.

## Gate

This ceiling passes only if all conditions hold:

1. aligned exact pointer plus replacement program `>=95%` overall and in both
   numeric and choice families;
2. aligned executed trajectory `>=95%`;
3. aligned clean idempotent copy `>=99%` and fault repair `>=90%`;
4. aligned exceeds hidden execution by at least 13 answers;
5. zero execution errors and zero decode-limit exhaustion;
6. copied-character rate `>=95%` and complete hash/latency/memory receipts.

A pass qualifies the frozen host's native value path for one explicit
pointer-policy integration. It is not itself a learned policy or a reasoning
claim. A fail closes this ceiling; no prefix, decoder-budget, checkpoint,
prompt, seed, or threshold variant is allowed.
