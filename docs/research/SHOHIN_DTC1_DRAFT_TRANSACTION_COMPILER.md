# DTC1: Draft Transaction Compiler

Status: development failed; exact DTC1 closed

Date: 2026-08-10

## Question

DTMC1 proved that the exact model-owned draft is causal, but a learned
fixed-slot decoder recovered only `45/666` answers. DTC1 tests a structurally
different interface: whether arithmetic transactions already serialized by
the immutable direct owner can be lowered by a generic grammar into a causal
typed program and recomputed by frozen learned LAM1.

This is a read-only interface and capability falsifier. It trains no weights.
It does not retry NMC1, TMC1, DTMC1, DSET1, or PSET1.

## Candidate

The only candidate source is the exact immutable direct development report
SHA-256 `234a029a...a49`. For every row, scan the complete draft from left to
right for balanced GSM arithmetic annotations of the form
`<<left_expression=claimed_result>>`. The last top-level equals sign separates
the expression from the claim. Reject an annotation rather than repairing it
when the delimiter, expression, result, or arithmetic domain is invalid.

The expression grammar is frozen to numeric constants, parentheses, unary
`+/-`, and binary `+`, `-`, `*`, `/`. Commas and a leading dollar sign are
removed from numeric atoms; Unicode multiplication/division are normalized to
`*` and `/`; a numeric percent atom means division by 100. Names, calls,
powers, comparisons, assignments, units, implicit multiplication, and every
other AST node are rejected. There is no natural-language semantic parser.

Each accepted expression is lowered postorder into the unchanged
`TypedMicrocodeGraph` instruction algebra. Numeric leaves resolve in this
fixed priority:

1. the most recent prior accepted transaction whose claimed result has the
   same exact rational value, yielding a causal `STATE` pointer;
2. every equal-valued numeric span in the source, yielding a `SOURCE` pointer;
3. otherwise an explicit `LITERAL`.

The claimed result is never executed, scored as a candidate answer, or copied
into a state value. It is only an episode-local alias key for a previous
computed state. Frozen learned LAM1 recomputes every accepted expression.
The graph commits the final accepted transaction state. Draft text outside
accepted annotations, including `####` answers, is ignored.

At inference there is no verifier, answer label, gold program, solver, host
repair, task router, or benchmark-specific route. The owner draft, generic
parser, typed graph, and learned arithmetic tables are the complete path.

## Immutable Inputs

- development data:
  `/lustre/fs1/home/sa305415/shohin/artifacts/product_reasoning/nmc1/data_bd30f2a_r1/development.jsonl`
  with SHA-256 `981b...` bound exactly by the evaluator;
- direct owner report:
  `/lustre/fs1/home/sa305415/shohin/artifacts/product_reasoning/nmc1/development_f24a237_r2/direct_normal.json`
  with SHA-256 `234a029a...a49`;
- direct owner reference: `267/666`;
- learned LAM1 checkpoint:
  `/lustre/fs1/home/sa305415/shohin/artifacts/product_reasoning/lam1/full_6b1c1f3_r1/microcode.pt`
  with SHA-256 `baab62ec...5c7c4`;
- public GSM8K test remains unopened.

The evaluator must bind the complete hashes, schemas, identity population,
owner checkpoint, generation seed, and decoding budget before parsing.

## Frozen Controls

All controls use the same 666 target identities and the deterministic
within-register-depth cyclic donor mapping already used by TMC1/DTMC1.

1. **Aligned:** target source plus its exact owner draft.
2. **Draft shuffled:** target source plus a depth-matched donor draft.
3. **Source plus draft shuffled:** donor source plus that donor's draft, scored
   against the untouched target answer.
4. **State reset:** aligned graph, but every causal `STATE` read receives zero.
5. **Opcode permutation:** aligned graph through the frozen LAM1 opcode
   permutation.

The last two are interventions on already compiled graphs, not alternative
parsers. Every parse failure, invalid graph, division by zero, and arithmetic
overflow fails closed.

## Prospective Metrics And Gates

Report annotation count, accepted transaction count, rows with at least one
accepted transaction, executable rows, answer exactness, linked rows,
state-read count, source-read count, literal-read count, invalidity by reason,
and per-row provenance. Also report direct-owner correctness crossed with DTC1
correctness so arithmetic repairs and semantic breaks are explicit.

Two claim levels are frozen:

### Interface qualification

All conditions are conjunctive:

- at least `500/666` rows contain an accepted transaction and execute;
- aligned accuracy is at least the immutable direct owner: `267/666`;
- aligned exceeds draft-shuffled by at least `100/666` answers;
- source-plus-draft shuffled is at most `67/666` answers;
- at least 100 aligned rows contain a causal state read;
- on aligned-correct linked rows, state reset loses at least 20 absolute
  percentage points;
- opcode permutation loses at least 30 absolute points from aligned; and
- no accepted aligned graph is invalid under normal execution.

### Capability improvement

DTC1 is an actual improvement only if interface qualification passes and
aligned reaches at least `280/666`, thirteen answers above the direct owner.
A lower result may identify a useful interface but cannot be called a
capability gain.

## Stop Rule

Run exactly one immutable development pass. A failure closes exact DTC1
without parser-rule, delimiter, normalization, ordering, threshold, donor,
or selection retries. The evaluator may be corrected only for a demonstrated
infrastructure defect before any score is produced. No public test opens from
an interface-only pass. A capability pass permits one separately frozen
public evaluation and a trainable transaction-policy successor; it does not
by itself establish general reasoning.

## Development Result

Immutable CPU job `750036` completed the single frozen pass in 0.41 seconds.
The exact report SHA-256 is `04a64643...467ce`; runtime commit `a7fae74` has
manifest SHA-256 `37a78599...0aa`.

Only `257/666` aligned drafts contain an accepted transaction and all 257
execute normally. The parser accepts 887 of 946 annotations, producing 674
causal state reads, 895 source reads, and 363 literal reads across 195 linked
rows. There are zero normal execution failures.

| View | Correct | Accuracy |
|---|---:|---:|
| aligned transactions | 108/666 | 16.2162% |
| depth-matched draft shuffle | 1/666 | 0.1502% |
| source plus draft shuffle | 1/666 | 0.1502% |
| immutable direct owner | 267/666 | 40.0901% |

The aligned path repairs seven direct-owner errors but breaks fourteen direct
answers. Of 98 aligned-correct linked rows, state reset retains only one, and
opcode permutation retains only four of all 108 aligned solves. Thus source,
transaction linkage, and learned arithmetic operations are strongly causal.

The gate nevertheless fails decisively: coverage is `257 < 500`, aligned is
`108 < 267`, and opcode loss is 15.62 points rather than the required 30.
The draft owner simply does not externalize any accepted transaction on 409
rows. This is not a parser-invalidity problem on admitted rows and cannot be
rescued by parser variants. Exact DTC1 closes; public GSM8K test remains
sealed. Compact result: `SHOHIN_DTC1_RESULT.json`.
