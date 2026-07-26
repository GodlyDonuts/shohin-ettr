# R12 ETTR G0 Horn Board Result

## Decision

`horn_version_space_falsifier_passes_candidate_reasoning_unproven`

This validates the first offline falsifier for the Endogenous Typed Theory
Reactor. It is not evidence that Shohin already reasons.

## Exact Mechanics

The board defines 20 theories, each containing three rules selected from a
six-rule typed Horn library. The universe contains two object types, five
unary or asymmetric binary predicates, six objects, and 27 well-typed ground
atoms.

The challenge set contains all 27 singleton initial fact sets and all 351
two-fact sets, for 378 states. Every theory has a unique behavioral signature.
Two independently implemented closure engines agree on all:

```text
20 theories x 378 challenges = 7,560 exact agreements
```

Greedy evidence selection needs one demonstration for seven theories and two
for thirteen. Exact version-space construction produces:

| Evidence class | Episodes | Behavioral classes | Required action |
|---|---:|---:|---|
| Singleton | 20 | exactly 1 | commit |
| Ambiguous | 20 | 2--20 | abstain |
| Contradictory | 20 | exactly 0 | reject |
| Coherent alternate | 20 | exactly 1, target excluded | commit alternate |

Four opaque renderers produce different source bytes while preserving the
same evidence and receipt. Reference encodings use only the generic
transaction substrate and require 19--23 cells, 23--34 edges, and 55--71
transactions.

Focused verification is nine passing tests, clean Ruff, and clean byte
compilation.

## Boundary

This board is assessor infrastructure. The candidate may not import its rule
library, exact engine, version-space enumeration, or semantic renderer state.
Typed rewriting and guarded-resource boards, process custody, synthetic
architecture training, and causal controls remain future work.

The architecture itself now exists separately in
`train/endogenous_typed_theory_reactor.py`; its readiness and parameter receipt
do not change this board's no-capability boundary.
