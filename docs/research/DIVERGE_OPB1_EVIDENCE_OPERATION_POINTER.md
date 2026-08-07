# DIVERGE-OPB1: Evidence Operation Pointer

Status: frozen before data materialization, training, or neural score.

## Capability Hypothesis

DIVERGE-SNL1 confirmed the composition of model-owned spanless value events,
neural unseen-law synthesis, natural command decoding, recurrent execution,
and late query readout. Its largest remaining compiler shortcut is an exact
whole-word search that assigns each evidence statement to one of eight
episode-local operation aliases.

OPB1 changes only that interface:

> A shared byte encoder and dynamic alias-table encoder can select the one
> operation named by a raw evidence statement, preserving SNL1's complete
> end-to-end execution under fresh opaque renaming and coherent table
> permutation without exact string search.

All qualified OQB1, NCP1, SVE1, NLS1, and executor weights remain
bit-identical. OPB1 does not retrain or fine-tune them.

## Candidate Mechanism

For source-token states `h_t` and episode-local alias states `a_j`, OPB1
computes normalized projected compatibility plus a learned source-token gate:

`e_tj = exp(s) * cos(W_h h_t, W_a a_j) + g(h_t)`

and commits one operation position with:

`argmax_j logsumexp_t e_tj`.

The runtime receives only raw evidence bytes and the eight-entry alias table.
It contains no regex search, substring search, alias index lookup, operation
label, numeric parser, or support solver. The selected hard positions group
the already-qualified SVE1 value events before the frozen NLS1 synthesizer.

## Frozen Data And Training

- 100,000 deterministic training rows, seed `2026080861`;
- development seed `2026080862`, 256 episodes;
- conditional confirmations `2026080863`--`2026080867`, 256 episodes each;
- 1,000 AdamW updates, batch size 128, learning rate 0.001;
- treatment and independently initialized-identical decoy-table control;
- fresh names, sources, and episode identities disjoint from the complete
  SNL1 lineage;
- operation targets and scrub aliases exist only in the assessor/training
  supervisor, never in candidate-visible evaluation records.

The decoy arm receives eight fresh names unrelated to the operation present in
the sentence while retaining the treatment target labels, update order,
initialization, parameter count, and compute.

## Frozen Arms And Gate

Positive arms are normal source/table, unseen full operation-plus-register
rename, and coherent operation-table reindex. Each requires:

- operation binding at least 99%;
- evidence, initialization, query, and law packets at least 99%;
- terminal states and answers at least 99%;
- at least 95% terminal-state exactness at every held depth;
- all frozen NCP1 program paths at least 99%.

Causal controls are fixed:

- cross-owner operation-table reindex preserves local binding at at least 99%
  but must reduce states to at most 5% and answers to at most 10%;
- replacing the named operation by an unseen out-of-table name must reduce
  operation accuracy to at most 20% and states to at most 5%;
- the matched decoy-table model must remain at most 20% operation accuracy and
  at most 5% state accuracy.

The development gate is conjunctive. A miss closes this exact mechanism
without width, seed, duration, threshold, renderer, or loss variants. A pass
opens the five fixed source-disjoint confirmations once; all five must pass.

## Claim Boundary

A confirmed pass would remove exact evidence-to-operation string binding from
the controlled SNL1 trajectory. Exact repeated-register occurrence
quotienting, the bounded 25-row coefficient vocabulary, and modular Z/97
execution would remain engineered. This is an architecture-mechanism result,
not yet an open-domain or public-benchmark reasoning claim.
