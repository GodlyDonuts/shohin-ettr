# R12 Sparse-Law Generator Factorization Preregistration

## Trigger

The first direct set-attention completer (`704748`) is a clean negative:

- training transition accuracy: 91.4150%;
- development transition accuracy: 46.5000%;
- development complete maps: 0/60;
- development exact queries: 4/60; and
- direction-negated exact queries: 8/60.

The training pool already covered 184 unique fitting action maps while all 80
development action maps remained hash-disjoint. More rows from the same pool
cannot test the missing generalization.

## New Hypothesis

Unseen operators may generalize if represented as compositions of reusable
learned permutation generators rather than decoded directly as independent
table cells.

The treatment adds:

- 32 learned soft permutation generators for each cardinality;
- eight-step Sinkhorn normalization for every generator;
- an observation-set encoder that selects four generator stages; and
- differentiable matrix composition to produce the complete operator.

No generator has a predefined arithmetic, bitwise, Gray-code, or family
meaning. The candidate does not enumerate the legal hypothesis union or call
the exact compiler.

## Frozen Canary

- board and map split: unchanged sparse-law V1;
- seed: 20260725;
- width: 128;
- recurrent byte layers: 2;
- generator count: 32;
- composition depth: 4;
- optimizer updates: 2,000;
- auxiliary fitting rows: 3,000;
- frozen fitting rows: 60;
- counterfactual direction rows: 120;
- batch size: 64;
- same-weight controls: direction negation, target shift, observation zero;
- candidate-time oracle/search/verifier calls: 0/0/0; and
- one H100, maximum one hour.

The exact held-out passive source order remains absent. Counterfactual rows
use its relation words only in different source/target orders.

## Decision Rule

This is a one-seed architecture canary, not promotion. Continue to the frozen
five-seed gate only if it:

1. exceeds the direct attention baseline's 46.5% development transition
   accuracy;
2. produces at least one exact unseen development map;
3. beats every same-weight control in query exactness; and
4. retains zero train/development action-map overlap.

Failure closes this learned-generator formulation. It does not close sparse
law induction generally.
