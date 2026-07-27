# R12 ETTR-IL-v3 Materialization Contract

**Contract ID:** `R12-ETTR-IL-v3-materialization-v1`

**Status:** implemented before architecture-facing v3 materialization

**Purpose:** preserve the broad initializer population frozen by
`R12-ETTR-IL-v3-initializer` without weakening the ETTR causal packet
contract or silently restoring the rejected v2 checkerboard filter.

## 1. Scope

This contract changes no neural module, tensor width, parameter count,
checkpoint, optimizer state, or training authorization. The complete Shohin
ETTR system remains 192,779,435 parameters. It changes only the validation,
objective, and receipt semantics used when broad v3 causal rectangles are
materialized.

The candidate protocol was frozen before this receiving-contract mismatch was
found. This document is therefore an explicit materialization-contract repair,
not a claim that the original candidate freeze already contained the repair.

## 2. Defect

The v3 protocol deliberately admits broad compiler, transaction, and
composition episodes. WORLD or COMMAND interventions in those episodes must
change the terminal packet, but they need not change the answer to every
query. The inherited v2 receiver instead required every edge of every
2-by-2 WORLD-by-COMMAND rectangle to change the query label.

That requirement would reject broad v3 rows after selection and recreate the
same sparse XOR/checkerboard population that v3 was introduced to avoid. It
would also train an impossible contrastive target on a valid no-effect query
edge.

## 3. Factorial geometry

Every architecture-facing causal rectangle still must:

1. partition its batch exactly into `(W0,C0)`, `(W0,C1)`, `(W1,C0)`,
   `(W1,C1)`;
2. keep the query prefix and autonomous read position matched;
3. use distinct WORLD and COMMAND source renderings on their declared axes;
4. keep packet support geometry fixed across the rectangle;
5. change the initial packet across the WORLD axis;
6. change the terminal packet on every WORLD and COMMAND edge; and
7. carry independently executed targets for all four corners.

An equal answer label is permitted only when those packet-level causal gates
pass. It therefore means that the selected query is invariant to a real state
change, not that the intervention was ineffective.

Strict answer-label checkerboards remain mandatory for separately designated
qualification/evaluation boards. They are not a universal initializer-data
shape and cannot be inferred from the generic rectangle type.

## 4. Query objective

For every matched intervention pair:

- cross-entropy supervises both factual answer labels;
- if `correct_target != foil_target`, the pair receives the original
  difference-in-differences causal margin;
- if `correct_target == foil_target`, the pair receives Jensen-Shannon
  invariance supervision between the two predictive distributions; and
- an effect pair can satisfy the margin only when its measured
  difference-in-differences reaches the configured margin.

The Jensen-Shannon term is finite, symmetric, and bounded. It does not create
an oracle route: the loss receives only the same immutable factual targets
already used for answer cross-entropy.

## 5. Receipts

Objective and optimizer-update receipts separately report:

- all classification-supervised query pairs;
- answer-changing contrast pairs;
- answer-preserving invariance pairs; and
- contrast pairs satisfying the causal margin.

For each WORLD and COMMAND arm:

```text
contrast_pairs + invariance_pairs = supervised_query_pairs
margin_satisfied <= contrast_pairs
```

These equalities are checked on device. The objective schema is
`shohin-ettr-composite-objective-v3`.

## 6. Acceptance

The repair is accepted only if:

1. every existing strict v2 objective and data-contract test still passes;
2. all-answer-changing rectangles retain their prior support and margin
   behavior;
3. all-answer-preserving rectangles produce finite differentiable loss,
   zero contrast support, and full invariance support;
4. mixed rectangles partition support exactly;
5. packet-no-consequence rectangles still fail closed; and
6. architecture-facing v3 materialization and its global audit bind this
   contract and exact source commit in their receipts.

This contract authorizes no model fitting.
