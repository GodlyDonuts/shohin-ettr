# DIVERGE-NVE1: Natural Variable Evidence Composition

Status: frozen before implementation, training, fresh-board materialization, or
result on 2026-08-06.

## 1. Purpose

NFE1 proves that a learned whole-mention interface can compile source-disjoint
integer evidence and drive exact delayed version-space commitment. It does not
test named-variable binding, rational values, guarded predicates, swaps, or
noncommuting state updates in the same natural-evidence path.

NVE1 is the one bounded composition gate for that missing interface. It keeps
the protected TOL3 source compiler, TFS1 factorized runtime, exact rational
executor, and verifier unchanged. It trains only one natural evidence compiler.
No Shohin weights, TOL3 weights, FTA1 weights, or TFS1 mechanics change.

## 2. Fresh board and evidence language

Generate exactly 256 fresh TFS-style episodes with seed `2026080611`. Each
episode contains five named rational registers, twelve binary semantic fault
lines, interleaved swaps, guarded predicates, and a late sensitive query. It
represents 4,096 coherent programs. Gold fault-line assignments and board
admission remain independent of every model score.

Replace each assessor-issued typed value receipt with one natural evidence
sentence. Every sentence contains:

- one one-based instruction ordinal;
- one source-owned target register;
- one distinct source-owned distractor register; and
- one signed integer or rational value.

The sentence asserts that the target, not the distractor, holds the value after
the instruction. Six renderer layouts are used only for training and three
different layouts only for confirmation. Confirmation layouts permute the
relative order of ordinal, value, target, and distractor. Full confirmation
sentences must have zero exact overlap with training sentences.

Training data contain 50,000 deterministic statements generated with seed
`2026080610`, the same source-owned register vocabulary, instruction ordinals
1--40, numerators -32--32 excluding zero where required, and denominators 1--7.
They are independent of confirmation episode states.

## 3. Learned evidence compiler

The compiler is a two-layer bidirectional byte GRU with width 192, no position
embeddings, and zero dropout. A lexical scanner proposes exactly two complete
numeric/rational mentions. The model assigns them to `STEP` and `VALUE` by one
hard permutation. The compiled program's symbol table proposes unique register
identity groups; repeated occurrences of the same register share one group.
The model assigns two distinct groups to `TARGET` and `DISTRACTOR` by one hard
permutation. It cannot create a register absent from the source-owned table.

Only after those hard assignments may exact code convert the one-based ordinal
to a packet step index and parse the signed rational. The compiled receipt binds
packet commitment, evidence sentence commitment, step, target-register
identity, distractor identity, and value. The distractor is provenance and must
not alter the asserted state predicate.

Training is one seed (`2026080610`), 1,000 AdamW updates, batch size 256,
learning rate 0.003 with cosine decay, class-balanced cross-entropy over both
hard role assignments, and the update-1,000 checkpoint only. No result may
alter width, duration, seed, optimizer, loss, renderer sets, or data quantity.

## 4. Composition and controls

The unchanged TOL3 compiler must first compile every fresh program and preserve
both options at all 3,072 fault lines. The natural evidence compiler then emits
the same typed equality receipts that TFS1 already verifies. Raw program and
evidence text, source residuals, and source KV state are unavailable after the
packet seal.

Matched arms are premature highest-support top-1, equal-memory complete
particles, no-evidence factorization, oracle typed evidence, and learned
natural evidence. Controls shuffle complete evidence sets across episodes,
swap target/distractor assignments, swap step/value assignments, reset initial
state, shift operation semantics, swap packet/query commitments, and poison raw
evidence text after sealing.

## 5. Frozen gates

All conditions are conjunctive:

1. TOL3 compiles 256/256 fresh programs, exactly two positive options at all
   3,072 fault lines, and 100% gold support;
2. at least 3,041/3,072 natural receipts have exact step, target, distractor,
   and rational value assignments, with every accepted receipt valid;
3. oracle typed evidence reaches 256/256 exact sensitive answers and exact
   factorized/enumerated extensional parity;
4. learned natural evidence reaches at least 245/256 exact answers and at least
   95% exact conditional on initially wrong top-1;
5. learned evidence beats top-1 and equal-memory particles by at least 50
   percentage points each;
6. no-evidence support abstains on at least 245/256 sensitive queries;
7. shuffled evidence, target/distractor swap, step/value swap, state reset, and
   operation shift each reduce exactness by at least 50 points; all packet/query
   swaps reject and post-seal evidence poisoning is bit-invariant;
8. zero wrong-source, wrong-step, wrong-register, wrong-value, or wrong-
   distractor receipt is accepted; zero false commitment, malformed accepted
   packet, gold-support deletion, or overflow; and
9. model parameters, training bytes, board/evidence hashes, canonical bytes,
   logical/unique applications, wall time, and per-renderer/depth/operation
   results are reported.

Failure at the evidence-component gate closes this interface before learned
composition. Failure after the component pass closes NVE1 without variants. A
pass ends local natural-interface qualification and authorizes implementation
of the integrated trainable DIVERGE module; it still does not authorize
continuation pretraining or a general reasoning claim.
