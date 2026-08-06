# DIVERGE-TOL3: Position-Free Local Semantic Anchors

Status: qualified controlled-language front end on 2026-08-06.

TOL2 fixes document binding but misses its gate because the unchanged TOL1
clause-global heads make 277 top-level operation mistakes and 89 local guard
semantic mistakes. TOL3 keeps the TOL2 symbol table, relation graph, guard
partition, canonical bytecode, and exact executor. It replaces global semantic
pooling with one small learned byte-level encoder applied independently to:

- each candidate source word, classified as `NONE` or one operation anchor;
  and
- the source substring between predicate operands, classified as one of six
  comparisons.

Supervisor-only training extraction may identify the gold anchor in the TOL1
training board. Runtime receives only source strings and model logits. It may
not use the supervisor anchor dictionary, answer labels, generator metadata,
or execution search. Position-free scoring makes operation meaning invariant
to clause order by construction.

The one frozen fit uses seed `2026080505`, width 64, one bidirectional GRU
layer, full-batch class-balanced cross entropy over the deduplicated local
training snippets, AdamW at `3e-3`, and exactly 750 cosine-decayed updates.
At runtime, every source word is scored independently. The compiler selects
the single word with the largest positive non-`NONE` logit margin and passes
that model-owned span, rather than a fixed verb lookup, to the relational
argument decoder. Comparator phrases are extracted structurally between the
two typed predicate operands and scored by the same local encoder's separate
head.

The opened TOL1 OOD board is development only. TOL3 must reach at least 90%
exact semantic programs, answers, exact guard clauses, and rows with every
guard exact, with zero accepted malformed packets.
A miss kills this local-anchor design. A pass permits one fresh confirmation
renderer under the already frozen TOL2 thresholds. No width, duration, or seed
variants are authorized.

## Development result and confirmation contract

The one fit completes in 6.844 seconds on CPU with 28,109 parameters and
reaches 68/68 exact deduplicated snippets. Checkpoint SHA-256 is
`b8b9dfe54b7ab4a31a74739625b8650fa4ee93a41221ab5d82610ebc1c030328`.
On the opened 1,024-row TOL1 OOD board, TOL3 reaches 1,024 exact programs and
answers, all 2,506 guard clauses exact, and zero invalid rows. Development
report SHA-256 is
`b86187b2bad4f3f953acfc2aa9665ae69fa944404f488dc80b3c17269277b40e`.

This pass authorizes one fresh 1,024-row confirmation board at seed
`2026080506`. It uses disjoint register names, body depth 15--20, recombined
direct-action orders, and the previously unused
`otherwise FALSE; if PREDICATE, then TRUE` guard order. The unchanged
checkpoint must satisfy the same four 90% exactness conditions and accept no
malformed packet. No result-dependent renderer or decoder changes are
permitted after the board hash is materialized.

## Fresh confirmation result

The board materializes exactly 1,024 programs / 23,063 clauses at depths
15--20. Every row contains guards, swaps, register operands, and rational
values. Its register names are disjoint from both prior name banks and its
program identity overlap with all 25,536 earlier rows is zero. Board/report
SHA-256 values are
`36a5fb51f5129294fac4a6ea30cef22c4637d4ae79f633bbee65cec2b5735ed3` /
`714eac5590209cb4b71974bb6da80273fe4e44338987eff49a542e451b8772e5`.

The unchanged checkpoint passes every condition:

| Metric | Fresh confirmation |
|---|---:|
| exact semantic programs | 1,024 / 1,024 |
| exact answers | 1,024 / 1,024 |
| exact guard clauses | 3,663 / 3,663 |
| rows with every guard exact | 1,024 / 1,024 |
| exact top-level operations | 23,063 / 23,063 |
| invalid rows / accepted malformed packets | 0 / 0 |
| operation-shift answers | 0 / 1,024 |
| binding-derangement answers | 21 / 1,024 |
| state-reset answers | 0 / 1,024 |

Confirmation evaluation SHA-256 is
`2f5b1ca3c08f4b82f8e220713441014178da72a257b8fafb67b0887f28ad5700`.
All six artifacts are hash-identical in local disaster recovery and immutable
Newton path
`artifacts/reasoning/diverge_tol3_0dfaf53_r1`.

TOL3 is promoted only as a controlled typed-language compiler. It has learned
local operation/comparator semantics and composes them with document-owned
binding across new names, order, and depth. Its exact rational executor and
grammar partition remain engineered, so this is not unrestricted language or
general reasoning. No TOL3 seed, width, duration, or renderer variant follows.
The next gate must combine this source compiler with factorized language-
derived fault lines, delayed evidence, and one coherent DIVERGE commitment;
single top-1 parsing and full particles are protected controls.
