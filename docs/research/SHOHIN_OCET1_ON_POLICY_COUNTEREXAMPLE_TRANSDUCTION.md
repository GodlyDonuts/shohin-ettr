# OCET1: On-Policy Counterexample Edit Training

**Status:** prospectively frozen before proposal output, 2026-08-10  
**Scope:** source-disjoint training/development; holdout sealed

## Diagnosis and hypothesis

RIFT1's tied second pass is 99.84% syntactically valid and strongly
draft-causal, but marks KEEP on many wrong model-generated proposals. Existing
training used deterministic answer substitutions; evaluation asks the model to
judge its own natural errors. The objective is therefore off-policy.

OCET1 changes the learning distribution, not rank, seed, duration, prompt, or
decoder budget. The immutable aligned DSET-Q35 owner generates one mandatory
rewrite for every source-disjoint training presentation. The generic executor
materializes it. Training labels are then derived mechanically:

- exact executed proposal: KEEP;
- incorrect local executed proposal: REPLACE_LAST from its model-owned final
  surface to the independently verified final surface;
- malformed proposal: fail-closed to the original draft and its existing
  verified transaction.

The original textual draft and proposal feedback are removed before the next
attempt. Only source, executed model-owned proposal, and training-only exact
transaction remain. At inference, the system is the RIFT tied recurrent
transducer with no verifier, label, solver, or semantic host.

## Frozen stages

1. Generate exactly one immutable aligned proposal per 15,278 training row
   (7,639 sources) with the pinned DSET checkpoint, greedy decoding, and the
   FRET executor. Hash every proposal and execution.
2. Audit complete source/proposal/target retention, local edit derivability,
   source-disjointness, corruption-family balance, and zero holdout use.
3. Warm-start the exact 1,179,648-parameter DSET residual, reset optimizer, and
   run one 256-update on-policy fit. No router/expert/base changes.
4. Evaluate the tied two-transition architecture on the exact opened 1,908-row
   development board against DSET, ISET, RIFT, and a matched label-permuted
   on-policy control. Hidden draft remains the causal control.

Before capability output, exact gates will retain RIFT's minimums: `>=1,874`
overall, `>=220/256` choice, `>=945/954` clean, `>=859/954` fault, `+13` over
ISET and every newly trained matched control, `>=95%` valid scripts, and zero
exhaustion. One pass may open one sealed holdout. A miss closes OCET1 without
data-count, update, rank, layer, LR, seed, threshold, or recurrence variants.

## Claim boundary

OCET1 tests whether closed-loop learning on immutable self-generated
counterexamples makes a model-owned recurrent edit architecture operational.
DAgger, self-training, edit models, and verifier-derived training labels are
prior art; no primitive is claimed as novel. The possible contribution is the
causal same-family draft-to-edit fixed-point system and its matched controls.
