# DIVERGE-RRG1: Relational Role Grounder

Status: frozen after CCR1 closure and before RRG1 implementation, training-data
materialization, model initialization, or result.

## 1. Capability hypothesis

SOT1, SRP1, and CCR1 were trained on a perfectly confounded supervisor: each
surface renderer exposed only one TARGET/DISTRACTOR mention order. CCR1 fit all
100,000 labels while ignoring SELF/OTHER identity and failed exactly on held
surface families. The relevant question was therefore not identifiable from
the fit distribution.

RRG1 tests this hypothesis:

> Once role order is counterfactually complete within every lexical family,
> a length-free, single-encoding relational matcher can learn stable
> TARGET/DISTRACTOR semantics that transfer across EVIDENCE and QUERY.

This is not a CCR1 data retry. CCR1 encodes two candidate-conditioned copies
and scores SELF independently. RRG1 removes candidate identity before one
shared encoding and performs one joint role-to-mention assignment.

## 2. Frozen representation

RRG1 has one fresh REFERENT owner:

1. Every contiguous occurrence of either supplied mention group is collapsed
   before encoding to exactly one shared learned `MENTION` token. Entity bytes
   and original mention length are absent.
2. One two-layer bidirectional GRU of width 192 encodes the canonical sentence
   once.
3. Hidden states at the two anonymous mention tokens are pooled per mention
   group. One global sentence state and two learned semantic role-slot vectors
   (`TARGET`, `DISTRACTOR`) produce a complete 2-by-2 compatibility matrix.
4. The only legal output is one hard two-way permutation. Assignment scores
   use both role slots and both mentions jointly; no fieldwise or independent
   candidate decision is permitted.

The owner is shared by EVIDENCE and QUERY and starts from seed
`2026080625`, not from NVE1, SRP1, or CCR1. Qualified TOL3 WORLD and NVE1
numeric-EVIDENCE owners are immutable and hash-checked.

## 3. Counterfactually complete fit data

Before model initialization, derive one immutable training corpus from the
qualified 50,000 evidence and 50,000 query supervisors.

- Each semantic item receives a paired TARGET-first and DISTRACTOR-first
  realization inside the same lexical-anchor family.
- Lexical anchors, clause order, and mention order are sampled independently.
- Each stage therefore has 100,000 rows; every family has exactly balanced
  `(0,1)` and `(1,0)` role assignments.
- Entity names remain training-only, but RRG1 cannot observe their bytes or
  lengths after canonicalization.
- Exact source overlap with the opened SRP1 development board and unopened
  CCR1 confirmation board must be zero.
- The report must prove pair completeness, balanced roles per family, unique
  identities, valid source labels, and source hashes. No model score enters
  generation or selection.

The generated grammar may reuse semantic anchor words from training but may
not copy any development or confirmation sentence template verbatim.

## 4. Frozen fit

- exactly 2,000 updates;
- each update: 128 evidence plus 128 query rows;
- AdamW, LR `3e-3` cosine to zero, betas `(0.9,0.95)`, weight decay `0.01`;
- gradient clip `1.0`;
- equal-weight complete-permutation cross entropy;
- no answer, execution, state, comparator, development, confirmation, or PL1
  outcome enters fit.

Training evaluation must report exact accuracy overall and by stage, lexical
family, role order, and paired identity.

## 5. Admission and confirmation

The opened SRP1 board is development-only. Admission requires all three:

- QUERY at least `765/768`;
- EVIDENCE at least `3,070/3,072`;
- at least `255/256` fully sealed episodes.

Only on admission may RRG1 open the existing source-disjoint CCR1 board with
board SHA-256
`299237068f436ba33a68487b5300fcd724f8c98bd8bfe6b1916a4ebc7541ebf7`.
Fresh promotion requires every CCR1 semantic/end-to-end condition, replacing
the inapplicable SELF/OTHER controls with:

1. role-slot swap loses at least 90 query points;
2. deleting the learned MENTION marker loses at least 49 query points;
3. arbitrary entity renaming changes zero logits and zero assignments;
4. RRG1 improves frozen SRP1 by at least four exact query transactions and is
   no worse on evidence;
5. WORLD/EVIDENCE/sealing, mode/renderer floors, answer/abstention/parity,
   shuffled evidence, state reset, operation shift, transaction rejection,
   post-seal poison, and zero-integrity-error requirements remain unchanged.

## 6. Pass/kill boundary

A miss closes RRG1 without width, depth, seed, duration, family, grammar,
marker, warm-start, loss, optimizer, or threshold variants. A pass qualifies
one natural PL1 integration. It does not establish open-domain reasoning and
does not authorize continuation pretraining.
