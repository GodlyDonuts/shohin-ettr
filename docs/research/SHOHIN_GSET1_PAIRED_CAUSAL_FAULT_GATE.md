# GSET1: Paired Causal Fault Gate

**Status:** prospectively frozen before GSET1 model output  
**Date:** 2026-08-09  
**Host:** pinned Qwen3.6-35B-A3B  
**Scope:** source-disjoint development; holdout sealed

## Why this experiment exists

The trained Qwen DSET transfer executes 1,822/1,908 trajectories (95.49%) and
beats the causally hidden-draft arm by 648 answers. Its exact gate fails on the
choice family. Read-only attribution is unusually sharp: all 128 clean choice
rows KEEP correctly; 79/128 faulted choice rows incorrectly KEEP; every one of
the other 49 emits a correct replacement. Forcing the replacement action on
the 128 faulted choice rows yields 128/128 exact scripts and 128/128 exact
executed trajectories. The value generator is therefore qualified. The
remaining defect is the decision owner.

GSET1 changes the architecture rather than retrying DSET rank, duration, seed,
or prompt settings. It separates error detection from edit serialization:

1. The frozen DSET model encodes the complete source/draft presentation.
2. A small model-owned paired fault gate reads the final contextual state and
   emits KEEP or REPLACE.
3. KEEP deterministically copies the draft.
4. REPLACE is inserted as the causal first action token; the frozen DSET model
   generates the old/new surfaces, which a generic deterministic executor
   applies.

The final trajectory depends on the gate: forcing KEEP versus REPLACE changes
the actual generation path. No verifier, answer label, task router, or semantic
host repair exists at inference.

## Frozen architecture and objective

The DSET checkpoint is fixed at SHA-256
`166d8cbafd5fa5aa842a42c8856294436ed2704db3d51396a3554576399ea8bb`.
Its 1,179,648 residual parameters remain frozen. The gate is:

```text
LayerNorm(2048) -> Linear(2048, 256) -> SiLU -> Linear(256, 2)
```

It has 529,154 trainable parameters. Training units remain paired by source:
clean requires KEEP and the corresponding fault requires REPLACE. The loss is
an equal-weight sum of action cross-entropy and a margin-2 paired contrastive
loss on the actual KEEP/REPLACE logits. Training uses 1,024 updates, 256 pairs
per update, AdamW at 3e-4 with cosine decay. Cached states are permitted because
the Qwen base and DSET residual are immutable and hash-bound; no hidden state is
used across train/development identities.

## Matched arms

- **Aligned:** complete draft-visible state and correct paired labels.
- **Swapped:** identical aligned states, parameters, updates, and loss, but
  KEEP/REPLACE labels are swapped within every source pair.
- **Hidden:** identical parameters, updates, labels, and geometry, with draft
  keys causally hidden during both state extraction and replacement generation.

The already-completed forced-action diagnostic is attribution only and cannot
make GSET1 pass.

## Frozen development gate

All conditions are conjunctive:

- aligned action accuracy >=95% overall;
- aligned action accuracy >=95% in every corruption family;
- paired counterfactual consistency >=95%;
- aligned executed-trajectory exactness >=98%;
- clean copy >=99%;
- fault repair >=97%;
- aligned execution exceeds swapped and hidden by >=20 absolute points each;
- swapped and hidden action accuracy are each <=60%;
- no decode exhaustion;
- at most three literal old-surface execution errors, the known bounded DSET
  serialization boundary observed before this contract.

A conjunctive pass opens exactly one sealed holdout. A miss closes exact
GSET1-v0; there is no width, seed, duration, margin, layer, or prompt rescue.

## Claim boundary

A pass would establish a transferable, model-owned detect-then-edit temporal
revision mechanism on a current MoE host. It would not establish general
reasoning, public-benchmark improvement, or novelty of binary classifiers,
low-rank adapters, or edit scripts in isolation.
