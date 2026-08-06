# DIVERGE-TOL3: Position-Free Local Semantic Anchors

Status: frozen before training or development result on 2026-08-05.

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
