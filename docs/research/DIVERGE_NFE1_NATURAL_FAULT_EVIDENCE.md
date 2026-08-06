# DIVERGE-NFE1: Natural Equation Fault Evidence

Status: frozen before implementation, training, board materialization, or result
on 2026-08-06.

## 1. Purpose

TFS1 proves learned operation alternatives, exact factorized execution, delayed
recovery, and coherent abstention in a controlled typed language. Its evidence
is still an assessor-issued typed receipt. NFE1 tests the next distinct
boundary: can a learned whole-mention interface compile source-bound numeric
evidence from independently verified arithmetic reasoning text and drive the
same DIVERGE refinement mechanism?

NFE1 does not update TOL3, FTA1, Shohin, or the exact executor. It trains only
one equation mention-role model. It is not continuation pretraining or a public
reasoning benchmark.

## 2. Data

The only source corpus is the frozen V10 artifact
`artifacts/product_reasoning/data/v10_tokenbalanced_35m20c10s10p25t_4m_verified_r1.jsonl`
at SHA-256
`2461d6f70b44a142854d56c24e1fb42d600065e5788a2c4e055ba47b12696549`.

Training uses every deduplicated exact integer equation span from rows whose
source is `reasoning_gym_trace`: 2,179 unique equations covering addition,
subtraction, and multiplication. A span is admitted only when an independent
integer evaluator verifies its RHS.

Confirmation uses source-disjoint `augmented_gsm8k` rows. A row is eligible
only when it contains a complete chain of two to five verified integer
equations, its final RHS equals the answer field, every operation's three
type-correct alternatives produce distinct immediate values, magnitudes remain
below 3,000,000, and none of its exact equations occurs in training. This
leaves 112 rows before selection. Hash-sort all eligible rows, retain every
depth-three-or-greater row, then take the lowest-hash depth-two rows until the
board has exactly 96 episodes. The expected board contains 223 transactions.
No model score participates in board selection.

For each confirmation equation, deterministically rotate its visible operator
`+ -> - -> * -> +` while preserving the independently verified LHS, argument,
and RHS. The corrupted text is the candidate source. It does not reveal the
gold operation. The observed RHS remains evidence and is never passed to the
operation head after source sealing.

## 3. Learned interface

The mention parser is a two-layer bidirectional byte GRU with width 128 and no
position embeddings. A lexical scanner proposes maximal signed-integer spans;
the model assigns each complete mention to one of `LHS`, `ARGUMENT`, or `RHS`.
Assignment is a hard one-to-one permutation over whole mentions, never
independent byte argmax and never a pooled ten-field query bank. Exact integer
parsing occurs only after the model selects a complete span.

Training is one seed (`2026080608`), 1,000 AdamW updates, batch size 256,
learning rate 0.003 with cosine decay, zero dropout, and class-balanced
cross-entropy over mention roles. The first checkpoint at update 1,000 is the
only checkpoint. No confirmation score may alter width, duration, seed,
optimizer, loss, or renderer.

The frozen FTA1 checkpoint
`9321b78372d9926930d4de073d70e82c94e8360a69e09be695bab91b2e479f2d`
provides support scores over the three scalar operation classes. NFE1 retains
all three type-correct operations as one coherent categorical fault line; it
never mixes operation fields. The visible corrupted operation is expected to
make top-1 wrong. Candidate support may rank alternatives but may not remove a
type-correct operation before evidence.

## 4. Execution

Each episode begins from the first learned LHS. At every transaction, the
source-sealed packet applies the three candidate operations to each surviving
scalar state using the learned ARGUMENT span. The separately sealed learned
LHS/RHS evidence packet retains a lineage only when its current state equals
the observed LHS and its successor equals the observed RHS. Exact state groups
merge only identical complete states. The late query asks for the terminal
scalar value.

Raw response text, question text, answer labels, gold operations, model hidden
states, and source KV state are unavailable after sealing. The independent
assessor may enumerate worlds and check arithmetic but cannot repair a packet.

## 5. Controls and gates

Matched arms are premature highest-support top-1, equal-memory complete
particles, no-evidence factorized support, posterior answer aggregation, and
full hard-evidence factorization. Controls shuffle evidence across episodes,
swap packet/query commitments, reset the initial state, rotate operation
semantics, and poison raw source after sealing.

All conditions are conjunctive:

1. at least 221/223 complete confirmation equations have exact learned
   LHS/ARGUMENT/RHS mention assignment;
2. every accepted packet contains all three distinct type-correct operation
   candidates and therefore retains the verified gold operation;
3. exact factorized/independent extensional parity on every accepted episode;
4. at least 92/96 exact terminal answers for full NFE1 and at least 90% exact
   conditional on initially wrong top-1;
5. full NFE1 beats top-1 and equal-memory particles by at least 30 percentage
   points each;
6. no-evidence support abstains on at least 90/96 terminal queries;
7. shuffled evidence, initial-state reset, and operation shift each reduce
   exactness by at least 50 points; every packet/query swap rejects and
   post-seal source poisoning is bit-invariant;
8. zero evidence receipt accepted with a wrong source, step, mention, or value;
   zero false commitments, malformed accepted packets, or overflow; and
9. canonical packet/particle bytes, logical/unique execution applications,
   wall time, source-admission counts, and per-depth/per-operation results are
   reported even though this low-depth interface gate has no fixed resource
   promotion threshold.

Failure at the mention gate closes this interface before composition. Failure
after a component pass closes NFE1 without a second seed, width, duration,
renderer, or loss variant. A pass authorizes a broader natural program/evidence
compiler with variables and predicates; it does not authorize continuation
pretraining or a general reasoning claim.
