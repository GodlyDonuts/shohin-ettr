# DIVERGE-EWC1: Equivariant WORLD Compiler

Status: frozen before data generation, training, or model scoring.

Pre-score implementation note: the first deterministic build rejected one
depth-28 row because the original 512-byte tensor cap was too short. It wrote
no train, development, or confirmation file. The cap alone is corrected to
768 bytes before rebuilding; seeds, rows, renderer assignments, depths,
models, training schedule, evaluator, and gates remain unchanged.

The next pre-score overlap audit rejected revision 1 because one opaque alias
(`kaquyekeko`) collided between training and confirmation; source and identity
overlap were both zero. That rejected corpus is preserved. The deterministic
nonce length alone is increased from 10 to 14 letters for revision 2, before
any model training or score. Seeds, semantic examples, split assignments, and
all gates remain unchanged.

## Capability hypothesis

DIVERGE-NPL2 reaches `85.6104%` source-deleted late-query exactness, exactly
matching its oracle-semantic PL1 ceiling, but its WORLD transaction is still
produced by one exact regular expression. EWC1 changes only that boundary:

> A learned byte-level compiler can bind source values to episode-local
> register identities and select the ordered operation mentions from natural
> WORLD text, while exact permutation equivariance prevents renderer-local or
> absolute-name conventions.

NPL2, NVE1 EVIDENCE, EIC1 QUERY, the 64-scalar plastic policy, executor, and
verifier remain unchanged and hash-protected. EWC1 is not a continuation-
pretraining experiment.

## Model-owned boundary

A generic lexer exposes only:

- unsigned integer spans;
- exact occurrences of the eight already-declared operation aliases; and
- exact occurrences of the two already-declared register names.

The lexer does not know which numbers initialize state, which alias mentions
are executable rather than distractors, their operation meaning, or the final
answer. A two-layer bidirectional byte GRU must:

1. assign one distinct numeric mention to each declared register; and
2. classify every alias occurrence as executable or irrelevant.

Selected alias occurrences are emitted in source order and mapped to the
episode-local alias table. Register-value scores are pairwise functions of
numeric and register mention representations. Reordering the register table
therefore permutes the output rows exactly, with no learned absolute register
embedding in the treatment path. Alias occurrence scoring shares all weights
and pools same-identity occurrences without alias-index parameters.

The matched absolute-role control has the same encoder, heads, parameter
count, data, updates, seed, optimizer, and batch size. It replaces the two
source-derived register keys with two learned absolute keys. This preserves
normal capacity but removes exact register-order equivariance.

## Data split

The frozen generator contains eight initial-state clauses and eight ordered-
program clauses. Their 64 Cartesian pairs are assigned by a fixed arithmetic
partition to train, development, or confirmation. Every phrase family occurs
across the campaign, but development and confirmation hold out complete
clause compositions. Clause order, distractor placement, initial values,
alias sequences, depths, aliases, and register names vary independently.

- training: 50,000 rows, seed `2026080721`, depths 3--20;
- development: 4,096 rows, seed `2026080722`, depths 3--28;
- confirmation: 4,096 rows, seed `2026080723`, depths 3--28.

Confirmation bytes are generated and overlap-audited before model scoring but
remain unopened unless the single development gate passes. Exact source,
identity, alias, and register overlap between every pair of splits must be
zero.

## Frozen controls

Evaluation reports complete typed-WORLD exactness normally and under:

- register-table transposition, mapped back to the original physical frame;
- a fixed eight-alias table permutation, mapped back;
- deterministic unseen entity renaming;
- context scrub preserving only candidate numbers and declared symbols; and
- the equal-parameter absolute-role control.

The model never receives `initial_state`, `symbols`, `numeric_targets`, or
`operation_targets` in its forward call. Those fields are assessor-only.

## Development gate

The one treatment and one matched control are trained for exactly 1,000
updates, batch 256, AdamW learning rate `0.003`, cosine decay, and seed
`2026080721`. Development passes only if all conditions hold:

1. treatment train typed-WORLD exactness is at least 99%;
2. normal joint exactness is at least 99%, initial state at least 99.5%, and
   operation sequence at least 99%;
3. every held-out renderer pair is at least 95%;
4. mapped register-order, mapped alias-order, and unseen-rename exactness are
   each at least 99%;
5. source scrub joint exactness is at most 20%;
6. treatment/control parameters, data, seed, schedule, and optimizer match;
7. control train exactness is at least 99% and normal development exactness is
   at least 95%; and
8. treatment mapped register-order exactness exceeds control by at least 25
   percentage points.

A development pass opens the frozen confirmation exactly once. Confirmation
must independently satisfy conditions 1--5 and all receipts. A miss closes
this exact EWC1 mechanism without width, duration, seed, prompt, threshold,
renderer, loss, or optimizer variants.

## Conditional integration

Only a confirmed EWC1 checkpoint may replace `parse_program_surface` inside
the protected NPL2 runtime. The integration must compile every acquisition and
transfer WORLD before any hidden outcome is observed, delete source bytes,
and reuse the unchanged NVE1/EIC1/PL1 path. It passes only if WORLD semantics
remain at least 99.5%, late-query exactness remains at least 80%, stays within
five points of oracle PL1, and all existing NPL2 destructive controls remain
qualified.

Even a complete pass establishes controlled learned natural-WORLD reasoning,
not unrestricted prose, public benchmark reasoning, or authorization for a
long pretraining run. The exact executor and verifier remain the next
engineered boundary.
