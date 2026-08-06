# DIVERGE-SRP1: Semantic Referent Primitive

Status: frozen after the SOT1 result and its one read-only attribution, before
SRP1 data materialization, training, or neural result.

## 1. Capability hypothesis

SOT1 assigned `TARGET/DISTRACTOR` semantics to lifecycle stages. Its isolated
QUERY owner fit all 50,000 training examples but inverted one fresh renderer
on every counterfactual row. The already-qualified EVIDENCE symbol head also
fails as a direct query reader. The defect is therefore not solved by choosing
one existing stage owner.

SRP1 changes the architectural ownership axis:

> A semantic role is owned once and reused across stages. EVIDENCE and QUERY
> may have different transaction schemas, but both must obtain referent roles
> from one exchange-equivariant REFERENT primitive.

This is not an SOT1 renderer, width, duration, seed, or loss retry. The SOT1
QUERY owner is a frozen baseline only.

## 2. Architecture

The composite contains three owners:

1. immutable TOL3 WORLD owner;
2. immutable NVE1 numeric-evidence owner for `STEP/VALUE` only; and
3. one plastic REFERENT owner shared by natural EVIDENCE and natural QUERY.

The REFERENT owner encodes source bytes and pools each complete symbol group.
For candidate group `i` and the other group `j`, one shared scorer computes

```text
s_i = f(global, h_i, h_j, h_i - h_j, h_i * h_j)
delta = s_0 - s_1
```

and emits the only two legal complete permutations from `delta`. Swapping the
two candidate groups therefore swaps the prediction by construction. No
stage-specific target head, independent field classification, host repair, or
fieldwise averaging exists.

The REFERENT byte encoder is initialized from the qualified NVE1 encoder, then
trained jointly on the immutable 50,000-row natural-evidence corpus and the
immutable 50,000-row natural-query corpus. The NVE1 numeric owner remains a
separate frozen copy; REFERENT plasticity cannot change numeric assignments.

## 3. Frozen fit

- seed: `2026080621`;
- updates: `1,000`;
- batch: `128` evidence plus `128` query examples per update;
- optimizer: AdamW, LR `3e-3` cosine to zero, betas `(0.9, 0.95)`, weight decay
  `0.01`, gradient clip `1.0`;
- loss: equal-weight symbol-role cross entropy for the two stages;
- no answer, state, operation, comparator, or confirmation label enters fit.

Qualified WORLD and numeric-evidence hashes must remain bit-identical. The
checkpoint serializes one model and one owner manifest.

## 4. Fresh board and controls

One 256-episode board at seed `2026080620` uses a disjoint 32-name entity bank,
new program identities, and six source-disjoint query compositions built from
training-supported semantic primitives. Every query mode crosses every
renderer; mode and renderer cannot be confounded. Exact source, query,
identity, and entity overlap with SOT1 confirmation is zero.

Matched controls are:

- frozen SOT1 isolated QUERY owner on the same fresh query transactions;
- frozen NVE1 EVIDENCE symbol head on the same transactions;
- typed-query ceiling;
- top-1 and equal-memory complete particles;
- referent-role swap, shuffled evidence, state reset, operation shift,
  packet/query swap, and post-seal poison.

## 5. Frozen pass/kill gate

All conditions are conjunctive:

1. WORLD programs at least `250/256` and natural EVIDENCE at least
   `3,041/3,072`;
2. natural QUERY at least `752/768`, every mode at least `245/256`, and every
   renderer at least `122/128`;
3. sensitive answers at least `245/256`, with exact extensional parity and at
   least 95% exact conditional on initially wrong top-1;
4. no-evidence sensitive abstention, invariant answers, and partial-evidence
   underdetermined abstention each at least `245/256`;
5. QUERY role swap, shuffled evidence, and state reset each lose at least 50
   points; packet/query swaps all reject and post-seal poison is invariant;
6. zero invalid transaction, false commitment, malformed packet, gold
   deletion, overflow, or valid-execution rejection;
7. immutable owner hashes remain unchanged; and
8. SRP1 improves exact query transactions over frozen SOT1 by at least
   `77/768` (10 points). If frozen SOT1 already passes or trails by less than
   ten points, semantic sharing has not earned inclusion.

A miss closes this exact semantic-primitive design. Do not run SRP1 width,
duration, seed, renderer, optimizer, warm-start, or loss variants. A pass
qualifies this referent interface for one broader end-to-end successor; it is
not open-domain reasoning or authorization for continuation pretraining.

