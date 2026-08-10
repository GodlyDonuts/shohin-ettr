# WGP1: Weighted Grammar Projection

Status: frozen no-training successor; development evaluation pending

Date: 2026-08-10

Parent: frozen BTT1 checkpoint
`2283c86b1a640c9d5c02ffbc70b4646a0a1ad3538c4e1d4ab4ea3b164363e278`

Holdout: sealed

## Hypothesis

BTT1 is exact on `98.8512%` of the source-disjoint board, and every one of its
45 errors is an invalid byte-role sequence. WGP1 changes no weights or logits.
It uses a fixed width-64 beam to select the maximum-log-probability complete
path satisfying only the generic expression grammar:

- numeric bytes are contiguous and contain at least one digit and at most one
  decimal point;
- operands and binary operators alternate;
- unary negation occurs only where an operand is expected;
- parentheses balance and close only after an operand;
- exactly one nonempty expression is selected from the raw tape.

The projection sees frozen per-byte role logits and raw bytes only to validate
copied numeric characters. It does not inspect an answer, target, verifier,
family, source template, or arithmetic value. The unchanged generic
shunting-yard executor materializes the projected roles.

## Frozen controls and gate

Run normal, same-family/depth source shuffle, zero-byte input, and flat
execution under the same frozen checkpoint and width-64 projection. Compare
row-wise against immutable BTT1 top-1 output.

The zero-byte arm may memoize exact projection results by tape length and the
per-position digit/dot/other mask. Under zeroed inputs, logits are identical
for equal lengths and grammar transitions inspect raw bytes only through that
mask, so this is exact execution reuse rather than an approximation.

The conjunctive development gate requires:

- complete and selected role exact `>=99.5%`;
- valid program `=100%`;
- exact skeleton `>=99.5%` and every family `>=99%`;
- mixed precedence, unary groups, and three-plus-parenthesis each `>=95%`;
- at least 30 of BTT1's 45 failures repaired and at most two previously exact
  rows broken;
- source shuffle and zero-byte controls each `<=25%`, with normal margins
  `>=74` points;
- flat execution loses `>=35` points on hierarchical rows;
- zero search exhaustion and no fallback to top-1.

One pass opens exactly one sealed holdout. Failure closes WGP1 without beam,
grammar, score, checkpoint, seed, or threshold variants. A pass establishes a
qualified source-to-program compiler, not learned arithmetic execution or a
novel parsing algorithm.
