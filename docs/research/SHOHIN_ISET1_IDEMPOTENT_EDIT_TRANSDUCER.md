# ISET1: Idempotent Span-Edit Transducer

**Status:** prospectively frozen before ISET1 model output  
**Date:** 2026-08-09  
**Host:** pinned Qwen3.6-35B-A3B  
**Scope:** source-disjoint development; holdout sealed

## Thesis

DSET's replacement value generator is qualified: externally selecting
REPLACE repairs all 128/128 faulted choice examples. DSET and GSET fail because
they first classify whether a draft is wrong. ISET1 removes that latent binary
decision. Every invocation emits one complete replacement transaction:

- clean draft: `REPLACE_LAST(old, old)`, an identity transaction;
- faulted draft: `REPLACE_LAST(old, corrected)`, a corrective transaction.

The same strict deterministic executor applies both. The architecture therefore
learns one sequence-transduction problem instead of a detector followed by a
conditional generator. A valid final trajectory is draft-dependent because the
emitted old surface must be present in the draft; there is no source-only
fallback, verifier, semantic parser, answer label, or task router at inference.

## Frozen experiment

The host, final-16 rank-18 post-MLP residual, NF4/BF16 loading, 1,179,648
trainables, 256 updates, LR 5e-5 cosine, four paired sources per update,
4,096-token custody, Qwen tokenizer, and source-disjoint identities are exactly
the completed DSET-Q35 transfer settings. The only changed factor is the edit
language and target objective.

Arms:

1. **Aligned:** clean identity transaction and fault corrective transaction.
2. **Swapped:** the two complete transactions are swapped within the same
   source pair, with identical geometry and compute.
3. **Hidden:** aligned transactions with draft keys causally unavailable.

The clean identity transaction is derived deterministically from the verified
fault transaction's corrected surface and is independently executed against
the clean draft before admission. Every train and diagnostic row must retain
its complete prompt and 32-token transaction.

## Frozen gate

All conditions are conjunctive:

- >=98% exact executed trajectories;
- >=95% exact scripts in every corruption family;
- >=99% clean identity edits;
- >=97% fault repair;
- >=95% paired transaction consistency;
- aligned execution exceeds swapped and hidden by >=20 absolute points each;
- zero execution errors and zero decode exhaustion.

A pass opens exactly one sealed holdout. A miss closes ISET1-v0 without
rank/seed/duration/layer/prompt variants.

## Claim boundary

A pass would establish an explicit, model-owned, host-agnostic draft-to-edit
transducer that transfers to a current MoE. It would not by itself establish
general reasoning or public-benchmark improvement.
