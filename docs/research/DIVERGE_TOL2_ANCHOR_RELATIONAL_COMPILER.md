# DIVERGE-TOL2: Document Anchor-Relational Compiler

Status: completed and missed its frozen development threshold on 2026-08-05.

## Hypothesis

TOL1 learned operation meaning but failed because every clause independently
classified arbitrary words into register roles. TOL2 keeps the exact TOL1
checkpoint and executor, adds no learned parameters, and changes only the
binding interface:

1. compile the leading declarations into one document-owned register table;
2. restrict every later register pointer to exact members of that table;
3. bind direct-action arguments by typed relations to a learned operation
   anchor rather than absolute source order;
4. split a guard into predicate, true action, and false action regions, score
   each action locally with the unchanged learned operation head, and classify
   the predicate through a canonical local guard;
5. canonicalize symmetric SWAP packets; and
6. execute one coherent exact-rational state lineage.

The relation grammar may use punctuation and closed structural words such as
`if`, `then`, `otherwise`, `to`, `from`, `by`, and `into`. It may not inspect
answers, execute alternate candidate programs, read generator metadata, or
introduce a lexical mapping from source verbs/comparators to opcodes. Opcode
and comparator identity must still come from TOL1 logits.

## Bounded gate

The already-opened TOL1 OOD board is development evidence only. With zero
updates, TOL2 must reach at least 90% exact executable answers, 90% semantic
programs, and 90% exact guards, while the preserved TOL1 answer remains
172/1,024. A miss kills the interface. A pass authorizes one source-disjoint
1,024-row confirmation board with a fourth renderer, new names, depths 15--20,
and a different guard ordering. Confirmation thresholds are frozen before its
generation: at least 85% answers and semantic programs, at least 80% guards,
and at least 50-point collapse under binding derangement and opcode shift.

This is a compiler-interface gate, not a claim of general language reasoning.

## Result

The zero-update interface raises exact OOD answers from TOL1's 172/1,024 to
763/1,024 (74.512%) and semantic complete programs to 749/1,024 (73.145%).
It therefore demonstrates that document-owned symbol restriction and
anchor-relative binding repair most of the failure, but it misses both 90%
promotion thresholds. No confirmation board is opened. Development artifact
SHA-256 is
`c8d3f112e18d96472f5be72fb38caf3d6435dc23f6ee62cc89c3647f7f1e4bea`.

The remaining errors are finite and localized. Top-level clause pooling makes
277 operation mistakes, predominantly 155 ADD and 111 SUBTRACT errors under
the held-out order. Local guard composition has 89 semantic errors: six true
branch opcodes, 39 false branch opcodes, and 44 comparators. After symbol-table
restriction, SET, QUERY, MULTIPLY, and canonical SWAP are effectively exact.

The immutable artifact's top-level `guard_clauses` counter records only
successfully decoded guards (2,461). The board and per-operation table give
the correct denominator, 2,506; later code corrects this reporting counter
without changing or rerunning the result.

TOL2 is closed. Its symbol-table, relation, guard-region, and canonicalization
mechanics are retained. The justified successor replaces clause-global
semantic pooling with one learned position-free classifier over local
operation-anchor words and comparator phrases. It must learn those labels from
the training board; a runtime lexical opcode dictionary remains forbidden.
