# R12 Token-Native Syntax-Graph Schedule Preregistration

## Causal Question

Does the sticky exact-executor path fail because a sequence encoder must infer
the token-native grammar and opaque-reference topology before it can compile a
coherent program?

The occurrence-linked arm rejects equality pooling alone. Its source and
target fields regress sharply even though identifier equality is supplied.
This treatment exposes the missing public structure directly: exact AST
parent/child edges and semantic child roles under all four renderers.

## Treatment

`TokenNativeSyntaxGraphEncoder` reconstructs, using only public token IDs and
arities:

1. the exact leading AST span, deleting deterministic transport cover;
2. one parent for every non-root syntax node under prefix or postfix order;
3. semantic child rank after reversing renderer-specific traversal order;
4. tree depth; and
5. equality edges between repeated opaque local identifiers.

Three learned graph layers exchange parent-to-child, child-to-parent, and
identifier-occurrence messages. The resulting memory feeds the existing
single sticky schedule compiler and unchanged exact typed-state algebra.

The treatment cannot inspect QUERY bytes, answer labels, target packets,
oracle programs, ontology sidecars, candidate scores, or host execution. A
one-to-one identifier renaming must leave every schedule distribution exactly
unchanged. Prefix/postfix and reversed-child parser tests must recover the
same semantic parent roles.

## Matched Gate

- protected checkpoint: `ckpt_0300000.pt`
- architecture/data seeds: `31/11`
- initial state: oracle
- start position: `0`
- updates: `1,000`
- learning rate: `3e-4`
- gradient clip: `1.0`
- scheduler: width `384`, 3 graph layers, 3 schedule layers, 8 heads
- pointers: ungrounded, matching control `725460`
- evaluation: unchanged 32-batch source-deleted four-arm board
- exact executor and typed query reader: unchanged

The scheduler has `13,482,273` trainable parameters and the complete system has
`169,421,167` parameters. The user's relaxed parameter limit is recorded, but
this gate does not rely on exceeding the old 200M ceiling.

## Decision Rule

Promotion requires simultaneous improvement in fully autonomous strict WORLD
and COMMAND over matched control without factual collapse. Field and exact-
schedule movement are diagnostic only.

- If source/target/value fields improve materially but strict pairs do not,
  syntax access is retained and the next arm uses one hard-selected latent
  program identity across every schedule field and step.
- If pointer fields remain flat or regress, explicit syntax routing is closed
  as a sufficient interface and the next intervention must alter compiler
  supervision or state representation rather than add width.
- Any positive strict result requires fresh architecture seeds and distinct
  held-out populations before promotion.

## Delayed-Generalization Test

A zero at 1,000 updates does not by itself exclude grokking. The architecture
will be measured at immutable 1k, 5k, and 15k budgets with the same seed,
ordering, objective, and evaluator. The relevant signature is not falling
training loss. It is:

1. near-saturation of training schedule imitation;
2. a delayed rise in held-out exact schedules rather than only marginal fields;
3. simultaneous delayed movement in strict WORLD and COMMAND; and
4. no decline in held-out factual top-1.

Smooth loss improvement, isolated value-field growth, or a late single-axis
spike does not count as grokking. The 15k matched schedule-only controls remain
the comparison, so longer compute cannot turn an ordinary budget effect into
an architectural claim.
