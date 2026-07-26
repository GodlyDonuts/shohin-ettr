# R12 Sparse-Law Neural Microcode Preregistration

## Trigger

Both architecture-generic table completers fail on hash-disjoint unseen
operators:

| Candidate | Development state accuracy | Complete maps | Exact queries |
|---|---:|---:|---:|
| Direct set attention (`704748`) | 46.5000% | 0/60 | 4/60 |
| Learned generator factorization (`704750`) | 15.7083% | 0/60 | 1/60 |

The factorized treatment also loses its direction-negated control in exact
queries. It is closed.

## Hypothesis

A 125M-scale language model may need an internal computational substrate
rather than being expected to rediscover arithmetic and bitwise execution in
its weights. The new treatment separates:

1. a learned byte controller that orients sparse demonstrations and predicts
   operator-family microcode plus parameters; and
2. a deterministic finite-domain ALU inside the model forward pass that
   executes the microcode and emits a transition distribution.

This is architecture-native execution: no host callback, parser, solver,
search, or posthoc verifier runs at candidate time.

## Fixed ALU

The ALU exposes three operation schemas over domains 8 and 16:

- modular affine;
- rotate/xor; and
- Gray-conjugated modular affine.

The controller predicts family, multiplier, offset, rotation, and mask
distributions. The ALU evaluates their differentiable mixture. Program
parameters in development are hash-disjoint from training.

Training receives preparation-only exact labels for the operation family and
its relevant parameters in addition to complete-map and source-direction
losses. Development receives no labels at candidate time; those labels are
used only for scoring. This treatment therefore tests supervised induction
into a fixed internal instruction set, not discovery of that instruction set.

This is deliberately an ontology-bearing architecture and cannot establish
open-ended law discovery. Its purpose is to test whether explicit internal
microcode closes the sparse-completion gap.

## Frozen Canary

The board, source deletion, map partition, optimization budget, and
same-weight controls are unchanged from the generator-factorization canary.
Four counterfactual source orders teach the held-out relation lexemes without
including the exact passive renderer.

Continue only if the treatment:

- exceeds 46.5% development transition accuracy;
- produces at least one complete unseen map;
- beats every same-weight control in exact query accuracy; and
- retains zero training/development action-map overlap.

A pass authorizes a five-seed microcode qualification, not a general-reasoning
claim.
