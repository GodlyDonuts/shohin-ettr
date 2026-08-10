# MLTC1: Monotonic Lexical Transduction Compiler

Status: frozen successor contract; CPU extensional audit pending

Date: 2026-08-10

Predecessor: PSTC1 closed at `91.8305%` exact complete skeleton

Holdout: sealed

## Hypothesis

PSTC1 learned a causal stack but its free-running controller lost source
position as programs became longer and scopes nested. MLTC1 removes global
action generation. A generic lexical candidate extractor exposes every
source-owned number span and every `+ - * / ( )` character in monotonically
increasing source order. A neural transducer assigns each candidate exactly
one role:

`IGNORE, NUMBER, NEGATE, ADD, SUB, MUL, DIV, LPAREN, RPAREN`.

A fixed shunting-yard executor consumes only those predicted roles and copied
number pointers, emitting the same typed `PUSH/NEGATE/APPLY/STOP` program used
by PSTC1. The executor cannot inspect an answer, call a verifier, repair a
prediction, or infer a role from source text. Raw source access is limited to
copying the span selected by a predicted `NUMBER` role.

This is structurally different from PSTC1: source traversal is monotonic and
one prediction is bound to one source candidate. The learned module no longer
has to remember which source symbol it should emit next. Scope is materialized
by the generic precedence stack over model-selected lexical roles.

## Frozen model and budget

- pinned frozen Qwen3.5-0.8B source encoder;
- one width-384, four-block bidirectional candidate encoder;
- source-span pooling, candidate-surface embedding, monotonic position
  embedding, and one role head with surface-valid hard masks;
- no recurrent action decoder and no learned arithmetic executor;
- exactly 1,024 updates, batch 32, AdamW LR `2e-4`, betas `(0.9,0.95)`, weight
  decay `0.01`, gradient clip 1, one seed;
- candidate-role cross entropy with `IGNORE` weighted `0.25` and every
  selected role weighted `1.0`;
- 32,768 charged examples, identical to FSTC1/PSTC1;
- fewer than 20M trainable parameters.

The independent CPU builder must reproduce every PSTC1 gold action program
exactly from gold lexical roles before a GPU fit opens. Any row mismatch is
fatal.

## Controls

1. same-family/binary-depth source shuffle under unchanged weights;
2. identical predicted roles with a flat left-to-right executor that removes
   precedence and parenthesis state;
3. within-batch candidate-state permutation while candidate surfaces and
   source positions remain fixed;
4. frozen PSTC1 and FSTC1 references.

## Development gate

The gate is conjunctive:

- lexical role sequence exact `>=99%`;
- selected lexical sequence exact `>=99%`;
- generic-executor valid program `>=99.5%`;
- exact materialized operation skeleton `>=97%`;
- every-family exact skeleton `>=95%`;
- mixed-precedence, unary-group, and three-plus-parenthesis exact skeleton each
  `>=90%`;
- source-shuffled exact skeleton `<=25%` and aligned margin `>=70` points;
- flat execution loses `>=35` points on hierarchical rows;
- candidate-state permutation loses `>=35` points overall;
- zero invalid source pointers, overlap, truncation, or decode fallback.

One development pass opens exactly one sealed holdout. Failure closes MLTC1
without width, depth, duration, seed, LR, role-vocabulary, tokenizer, loss, or
threshold variants. Arithmetic transition learning remains closed until this
compiler gate passes.

## Claim boundary

A pass establishes accurate model-owned lexical selection plus deterministic
hierarchical program materialization. It does not establish learned arithmetic
execution, broad language reasoning, or novelty of shunting-yard parsing.

## CPU admission

Job `749675` admitted all `75,935` training and all `3,917` development rows
with exact extensional parity to PSTC1. Maximum candidate counts are `36` and
`30`, below the frozen 64-slot bound. Train/development SHA-256 are
`33b8eb7bf154f3e938de00f06aebd7adb720eb4a1b6bd3f81554f9eb0b11bf3f`
and `8e867e7cdcc47015096979314450790cd86c804fd841b34d3df411a236b33679`;
report SHA-256 is
`8bcb025f376a2d3d481cbe8b00e878c09baa850e5f71683d424bae4e8a0d7e0f`.
