# Shohin ECTR0 Executor-Conditioned Temporal Revision

Status: frozen zero-training development screen; no output has been opened.

## Question

CTF1 showed two simultaneous facts on the same 666 source-disjoint GSM8K
development identities:

- the untouched Qwen3.5-4B owner produces a useful canonical transaction
  trajectory whose learned execution solves `419/666` rows causally; and
- forcing every answer through that trajectory loses capability relative to
  the owner's own direct answer (`487/666`).

ECTR0 tests the structurally different composition: preserve the complete
owner trajectory and learned execution as evidence for an already-qualified
later revision owner, then let that owner emit a new complete natural
solution. The learned executor is advisory evidence, not the final answer
path.

## Frozen Owners And Inputs

- Draft/transaction owner: exact immutable CTF1 normal report SHA-256
  `dc4e939b8186393ad6827f6cecbeacaaf86231d2abf2682ce093d43364b905f0`.
- Source board: exact CTE1 development SHA-256
  `aff466172c74dd7d13a183d117e32ec10d5da2048d10253189e4e3b3a599eb04`.
- Revision owner: qualified IDR4 Qwen3.5-4B checkpoint SHA-256
  `ae3847fe0728b1debcc13049822ea7499f744836b62d6d1c5bcb7c1000d8560b`.
- Backbone: `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Decoding: greedy, no thinking mode, 512 new tokens, batch four, seed
  `2026081061`, 4,096-token context, no training.
- Public GSM8K test and every other protected board remain sealed.

The revision prompt uses the qualified IDR source-plus-internal-draft format.
The internal draft contains the exact CTF1 owner completion. Depending on the
arm it then contains either the aligned learned-executor receipt, an unavailable
receipt marker, or a deterministic receipt from another row in the same
compile-status/register-depth stratum. No gold answer, verifier output, or
correctness label is visible.

## Arms

1. `aligned`: exact owner trace plus its own learned-executor receipt.
2. `receipt_absent`: exact owner trace plus `<RECEIPT_UNAVAILABLE>`.
3. `receipt_shuffled`: exact owner trace plus another identity's receipt,
   deterministically rotated within compile-status/register-depth strata.

All arms use identical source, owner completion, revision weights, generation
budget, evaluator, and identity order. Only the executor receipt differs.

## Prospective Gates

ECTR0 qualifies only if all conditions hold:

1. aligned revision reaches at least `500/666`, thirteen answers above the
   frozen direct-owner result;
2. aligned exceeds both receipt-absent and receipt-shuffled by at least 13
   answers;
3. relative to the direct owner's claimed final answer, aligned semantic
   repairs minus semantic breaks are at least 13;
4. at least 650 completions contain an explicit final answer;
5. every identity appears exactly once, no prompt is truncated, all protected
   hashes match, and all three arms complete under the same evaluator.

A miss is evidence that the qualified reviser does not already consume this
executor interface. It does not reopen CTE1/CTF1, alter their closed results,
or justify a nearby prompt/threshold/checkpoint retry. Receipt-specific
training may be considered only as a separately frozen mechanism with a
same-source counterfactual objective that makes the receipt identifiable.

The direct-owner reference uses CTF1's exact last `####` numeric claim. A
trailing number without that marker is not a completed direct answer.

## Claim Boundary

A pass would show that model-owned trajectory plus learned execution can
causally improve a trained same-family temporal reviser without changing any
weights. It would not prove that canonical transactions alone outperform
direct generation, nor that the executor is infallible, nor that the result
transfers beyond this source-disjoint development board.
