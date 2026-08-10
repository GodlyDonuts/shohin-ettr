# FSTC1: Fixed-Slot Typed Compiler

Status: closed development failure with strong causal signal; holdout sealed

Date: 2026-08-10

Predecessor: SLC1 closed (`1.2765%` terminal, `0/613` depth-five)

Holdout: sealed

## Capability hypothesis

SLC1 failed because one autoregressive text stream had to learn formatting,
program length, operation order, references, and exact arithmetic at once. Its
dominant behavior was a one-record collapse. FSTC1 replaces free-form ledger
generation with a bounded typed state transition:

\[
z_{t+1}=R(z_t,\operatorname{Attend}(z_t,H_x),e(\hat s_t)),\quad t=0,\ldots,4
\]

where each hard slot prediction is

\[
\hat s_t=(\text{ACTIVE/STOP},\text{op},r_L,p_L,r_R,p_R,v_t).
\]

`r` is either one of at most seven source-number spans or a causally prior
slot; `p` is identity or negation; and `v_t` is an exact signed-rational digit
state. The same recurrent cell and typed heads are reused at every slot. A
generic renderer may serialize predicted fields but cannot infer an operation,
execute arithmetic, repair a value, or inspect an answer.

The hypothesis is narrow: explicit typed recurrent state will learn program
unfolding and preserve causal state substantially better than SLC1's free-form
text decoder. Passing FSTC1 is a compiler qualification, not a claim of broad
reasoning.

## Admitted geometry

The immutable RG-v4 ledger has 75,935 train and 3,917 source-disjoint
development rows, depths one through five, and at most seven source numeric
spans. Questions are at most 174 characters. Exact state requires at most 23
numerator digits and 11 denominator digits in training (16/10 on development).

The first pointer audit exposed 3,231 train and 159 development operands that
the old ledger mislabeled as fresh literals. Every one is exactly the negation
of a causally prior result. Adding a polarity bit to prior-slot references makes
all `79,852/79,852` rows pointer-supervisable with zero unmatched operands.
This result is hash-bound by
`2fe9c48638f9ebd0ffbba41fa03f4a07817e2c64fe8ac2f1806f2213a7be18a9`.

Generic numeric-span discovery is lexical only. It may expose byte offsets and
decimal characters but may not parse expression structure, precedence,
operations, dependencies, or answers. Equivalent repeated source occurrences
are scored by value equivalence; exact occurrence identity is reported
separately.

## Architecture

The source stream is encoded once by a frozen language backbone plus a
trainable bidirectional source memory. The state stream is a tied recurrent
slot cell with cross-attention to that source memory. Source and state custody
remain separate; only attention reads source state.

- frozen host: pinned Qwen3.5-0.8B used by SLC1;
- source memory: width 512, four bidirectional Transformer blocks, eight heads;
- recurrent core: one tied GRU/cross-attention transition, width 512;
- heads: ACTIVE/STOP, four operations, left/right reference, left/right
  polarity, and typed result state;
- source references: seven lexical numeric spans plus masked padding;
- state references: five slots with a strict `< current slot` mask;
- result representation: sign, 23 numerator digits, 11 denominator digits,
  canonical length/EOS masks; zero and denominator constraints fail closed;
- hard autonomous feedback: straight-through categorical states during joint
  training and ordinary argmax during evaluation;
- expected trainable sidecar: below 30M parameters; exact receipt is required
  before a fit.

No host expression parser, calculator, symbolic executor, answer label,
verifier, or semantic repair is available at inference.

## Factorized gates

### A. Skeleton compiler

Train source memory, recurrence, STOP, operation, reference, and polarity heads.
Gold result-state embeddings may be supplied only as an explicitly labeled
teacher-forced input arm. The autonomous arm feeds predicted prior references
and predicted slot state. Result digits are not scored in this gate.

Development pass is conjunctive:

- depth exact `>=99%`;
- operation sequence exact `>=97%`;
- operand kind/value/polarity exact `>=97%` each;
- complete skeleton exact `>=90%`;
- every-family complete skeleton `>=85%`;
- depth-five complete skeleton `>=80%`;
- aligned-minus-source-shuffled complete skeleton `>=65` points;
- source-shuffled complete skeleton `<=25%`;
- recurrence-reset intervention loses at least 20 points at depth `>=3`;
- zero invalid forward references and zero decode exhaustion.

### B. Model-owned arithmetic transition

With oracle operation and exact operand digit states, a tied digit-circuit
transition predicts one exact canonical rational result. Data are generated
from the admitted value/operator distribution with a source-disjoint random
seed and an unopened wider/deeper confirmation partition. No arithmetic
routine runs inside inference.

Development pass is conjunctive:

- exact result `>=99.5%` overall and `>=99%` for every operation;
- sign, numerator length, denominator length each `>=99.9%`;
- exact result `>=98%` on the longest admitted digit bucket;
- shuffled operation or shuffled operand control `<=25%` exact;
- carry/borrow-state reset loses at least 30 points on affected examples;
- zero malformed rational states.

### C. Joint autonomous compiler

This opens only if A and B pass unchanged. It composes their qualified owners,
then permits one preregistered joint release. No gold slot or value enters the
autonomous path.

Development pass is conjunctive:

- valid typed program `>=99%`;
- exact complete records `>=90%`;
- terminal exact `>=90%` overall and every family;
- depth-five terminal exact `>=80%`;
- aligned-minus-source-shuffled terminal `>=65` points;
- source-shuffled terminal `<=25%`;
- autonomous-minus-teacher-forced terminal gap `<=10` points;
- zero invalid references, malformed values, or exhaustion.

One development pass opens exactly one sealed holdout. A component failure
closes that exact mechanism without rank, width, layer, duration, seed,
threshold, renderer, or prompt variants.

## Controls and accounting

The primary causal controls are source shuffle, recurrence reset, operation/
operand shuffle, and hard-state permutation. A parameter/FLOP-matched
independent-slot decoder is required before a joint claim; it shares source
memory but replaces tied recurrence with five independently queried slots.
SLC1 remains the frozen autoregressive reference.

Every report records source/data/checkpoint/runtime hashes, trainable and total
parameters, charged examples/tokens, forward/backward FLOPs, activation peak,
wall time, generated slots/digits, per-depth/family outcomes, and all control
interventions. Holdout remains unopened until the complete development gate
passes.

## Resource estimate

CPU tensorization and exact-reference audit: less than one CPU-hour. Mechanics:
one H100 for at most 15 minutes. Skeleton development: one H100, estimated two
to four hours. Arithmetic transition can run independently on a second H100,
estimated two to four hours. Joint release is not allocated until both pass.
Expected pre-joint charge is below eight H100-hours.

## Frozen result

Mechanics job `749647` passed both gold and hard recurrent feedback with finite
loss and gradients. Full fit `749648` completed 1,024 updates over 32,768
examples in 174.21 seconds (`188.09` examples/s), using 2.38 GB peak H100
allocation. The sidecar has exactly 21,816,330 trainable parameters; checkpoint
SHA-256 is
`13777429b2c01047e7514f201d325566c44d2df113f68dd54286da6fac5f759a`.

Parallel evaluations `749649--749651` closed the gate. Normal complete
skeleton accuracy is `3348/3917 = 85.4736%`; source-shuffled is `0/3917`, and
resetting recurrence removes `70.98` points on depth-three-or-greater rows.
Thus source and recurrent state are both causal, but the absolute gate fails.
Depth, operation sequence, reference kind, operand value, and polarity are
`95.7365%`, `90.6051%`, `93.6176%`, `89.4817%`, and `92.8772%`.
Depth-five complete accuracy is `66.0685%`; weakest-family complete accuracy is
`56.4677%`. No holdout opened and arithmetic transition/joint gates remain
closed.

The read-only attribution localizes the miss to syntax requiring hierarchical
scope. Products are `100%`, chain sums `97.88%`, and decimal chains `91.94%`.
Non-mixed expressions score `95.20%`, but mixed precedence scores `46.96%`.
Zero-parenthesis examples score `92.52%`, one-to-two parentheses `56.68%`, and
three-plus parentheses `14.14%`; unary-group expressions score `25.59%`.
This closes fixed-slot recurrence without nearby variants. The successor must
introduce an explicit model-owned parse stack/tree and scope transitions.
Comparison, training, and attribution SHA-256 are
`74035d8175145533b5a6d985c67e46564e73ced565fdca152cb7bcea17149cee`,
`bd9751bb1b32fb2164178d706653152314431f9a4c85e20bd1e950ae164f8b24`,
and `5b7f8d2c162950d8ddfb405ed56733630b0781bbb626ae588eea51b87faa62ea`.
