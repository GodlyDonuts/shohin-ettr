# PSTC1: Pushdown-Stack Typed Compiler

Status: frozen successor contract; CPU target admitted; mechanics pending

Date: 2026-08-10

Predecessor: FSTC1 closed at `85.4736%` complete skeleton

Holdout: sealed

## Hypothesis

FSTC1 is nearly exact on flat programs but fails predictably under nested
scope, unary groups, and mixed precedence. Increasing flat recurrence would
not represent the missing invariant. PSTC1 replaces its five independent
output slots with a model-owned pushdown transition over a small action
alphabet:

- `PUSH(source_number)`;
- `NEGATE`;
- `APPLY_ADD`, `APPLY_SUB`, `APPLY_MUL`, `APPLY_DIV`;
- `STOP`.

The recurrent controller attends a separately encoded raw source and the top
two stack states. Hard actions update a bounded tensor stack. `PUSH` copies one
model-selected source-number representation; `NEGATE` changes its typed
polarity; `APPLY` pops two states, writes one ordered operation record, and
pushes one result-owner state. A generic stack executor materializes only the
predicted actions and pointers. It cannot parse the source, choose precedence,
execute arithmetic, repair a program, inspect an answer, or invoke a verifier.

This is structurally different from FSTC1: hierarchy and scope are represented
by stack topology rather than inferred inside one flat slot state.

## Training-only supervision

An independent exact supervisor extracts the arithmetic expression from each
admitted family, parses it only during corpus construction, emits postorder
`PUSH/NEGATE/APPLY` actions, executes those actions with exact rational
arithmetic, and requires operation-by-operation parity with the immutable gold
ledger. The supervisor and exact values are absent at model inference.

Admission requires every train and development row to have:

- one unique source expression span;
- exact source ownership for every pushed number;
- exact action execution and terminal parity;
- exact ordered binary-operation/result parity with the existing ledger;
- bounded action length and stack depth;
- zero source overlap changes and zero holdout access.

Any row that cannot meet all conditions is excluded as a whole and reported.
The gate may proceed only if at least 99% of every development family remains.

## Frozen model and budget

- pinned frozen Qwen3.5-0.8B source encoder;
- width-512, four-block bidirectional source memory;
- one tied width-512 controller with source cross-attention and top-two stack
  reads;
- one shared stack-write cell and action/pointer heads;
- at most 22 actions and stack depth 6, fixed by the CPU receipt;
- 1,024 updates, batch 32, first 128 updates gold-action feedback and all
  remaining updates hard autonomous feedback;
- AdamW, LR `2e-4`, betas `(0.9,0.95)`, weight decay `0.01`, gradient clip 1;
- exactly the FSTC1 charged example budget (32,768) and one seed;
- trainable sidecar must remain below 30M parameters.

No FSTC1 weights are warm-started. This makes the changed state topology the
principal intervention.

## Controls

1. same-family/binary-depth source shuffle, paired by nearest action length;
2. stack reset before every action while retaining identical weights;
3. stack-top permutation at evaluation;
4. parameter/FLOP-matched recurrent decoder whose action state cannot read the
   stack;
5. frozen FSTC1 and SLC1 references.

## Development gate

The pass is conjunctive:

- action length exact `>=99%`;
- action sequence exact `>=97%`;
- source pointer value exact `>=97%`;
- generic-executor valid program `>=99%`;
- exact materialized operation skeleton `>=92%`;
- every-family exact skeleton `>=88%`;
- mixed-precedence exact skeleton `>=85%`;
- unary-group exact skeleton `>=80%`;
- three-plus-parenthesis exact skeleton `>=80%`;
- aligned-minus-source-shuffled exact skeleton `>=65` points;
- source-shuffled exact skeleton `<=25%`;
- stack reset loses at least 30 points on mixed/unary/parenthesized rows;
- stack-top permutation loses at least 20 points on depth-three-or-greater;
- no invalid pop, overflow, forward reference, or decode exhaustion.

One development pass opens exactly one sealed holdout. Failure closes PSTC1
without action-vocabulary, width, depth, duration, seed, LR, prompt, tokenizer,
or threshold variants. Arithmetic-state learning and record editing remain
closed until the compiler passes.

## Claim boundary

A PSTC1 pass would establish learned hierarchical source-to-program
compilation with a causal model-owned stack. It would not by itself establish
broad reasoning, arithmetic execution, or a novel universal architecture.

## Data admission result

CPU job `749652` admitted all `75,935` train and all `3,917` development rows
with zero exclusions. Every postorder action sequence reproduces the existing
binary operation and result sequence exactly. Maximum action count is 22 and
maximum stack depth is 6. Train/development SHA-256 are
`b522d58b3403064987a47d7b22657ef862945f18dfb352a78625fbcc27ebf008`
and `bf90044cd17e5e63cdeb7f91bffaae6e4bc59e992fb95cb8da9d2b29da00c914`;
report SHA-256 is
`31913bcd4fb83a7481221b56dc675ead4179e0f1bb8bec5eaa3d494e45f94b37`.
