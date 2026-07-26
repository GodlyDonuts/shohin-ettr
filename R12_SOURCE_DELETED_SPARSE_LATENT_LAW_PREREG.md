# R12 Source-Deleted Sparse Latent-Law Qualification

## Purpose

The variable-topology semantic compiler solved complete anonymous transition
tables. This successor gate removes that crutch. It asks a learned candidate
to infer complete operators from inclusion-minimal demonstrations and execute
only transitions that were absent from the source.

Passing this gate would establish learned sparse law completion across several
structured operator families. It would still not establish unrestricted
natural-language or general reasoning.

## Frozen Board

- operator families:
  - affine modular permutations;
  - bitwise rotate/xor permutations; and
  - Gray-code-conjugated affine permutations;
- cardinalities: 8 and 16;
- opaque action keys and explicit integer state values;
- action counts: 2 through 4;
- five fitting renderers and one held-out passive renderer;
- unseen-law, longer-composition, unseen-topology, held-out-renderer, and
  joint cells;
- 120 frozen episodes and 120 unique episode laws;
- only 812/4,200 = 19.3333% of complete transition records visible;
- all 387 query steps use unobserved action/state transitions;
- every retained record is necessary: deleting any one of the 812 records
  makes at least one action law non-identifiable;
- 97 individual action maps in fitting and 80 in development, with zero
  overlap;
- no family labels in candidate source or query; and
- one union hypothesis space containing 78 maps at cardinality 8 and 309 at
  cardinality 16.

The exact compiler enumerates that union only as an audited upper bound.
Candidate code may not import it, enumerate the legal maps, call a
family-specific formula, or use a board-family label.

## Candidate

The first candidate is a shared set-attention law compiler:

1. a byte encoder predicts source/target direction for each record;
2. numeric source/target pairs are grouped only by opaque action equality;
3. every possible source state attends over that action's sparse
   demonstrations;
4. one shared decoder predicts a complete transition row for every action;
5. sealing fails closed unless each predicted action is a permutation; and
6. a late query is executed only from the sealed packet.

Deterministic preprocessing may recognize the public domain header, decimal
integer literals, opaque action tokens, equality, line boundaries, and query
occurrence order. It may not parse renderer direction or infer a legal law.

## Causal Controls

All controls use the treatment weights with zero additional optimizer steps:

- negate predicted record direction;
- shift every oriented demonstration target by one modulo the domain; and
- zero all demonstration pair representations.

The final report must also include raw state-level transition accuracy,
complete-map exactness, valid-seal rate, query exactness, family/cell
breakdowns, and training/development action-map overlap.

## Promotion Gate

Across five independently generated boards:

- training/development action-map overlap: zero;
- candidate-time oracle/search/verifier calls: 0/0/0;
- treatment query exactness: at least 80% overall;
- treatment complete-map exactness: at least 80% overall;
- every family and development cell: at least 70% query exactness;
- treatment margin over every same-weight control: at least 10 points;
- every seed's treatment-control direction positive;
- source bytes absent from every deployed packet; and
- complete-system parameter count below 200,000,000.

A one-seed canary may diagnose optimization and architecture before the
five-seed gate. It cannot promote a reasoning claim.

## Scope Boundary

Even a pass would remain a bounded finite-domain result. The operator
hypothesis universe is finite, states are explicit integers, and execution is
a fixed discrete loop. The next boundary would be architecture-native Shohin
integration and transfer to a genuinely different law ontology. Continuation
pretraining remains held by user instruction.
