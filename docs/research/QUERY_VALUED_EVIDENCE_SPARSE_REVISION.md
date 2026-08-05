# Query-Valued Evidence Sparse Revision

Status: one bounded credit-assignment pilot; no reasoning or novelty claim.

## Hypothesis

CGSGR proved that counterexample-conditioned state revision is active, but it
lost to fixed evidence coverage because contradiction magnitude is not answer
utility. Query-Valued Evidence Sparse Revision (QVESR) keeps the same coherent
state, consequence model, sparse write budget, and recurrent revision
operator. Its only substantive change is a learned value function over each
source-owned evidence item:

```text
u_j = Value(S_t, evidence_j, residual_j, query, visit_count_j)
J_t = HardTopK(u)
S_(t+1) = SparseRevise(S_t, evidence[J_t])
```

The forward pass is hard top-k and admits exactly two evidence items per
round. A straight-through probability path lets final-answer loss train the
selector. No target answer, hidden state, tool, teacher, or external verifier
is available to the selector at inference. The query is intentionally
available during deliberation because the failed hypothesis is precisely that
query-agnostic local error predicts answer value.

## Prior-art boundary

Learned attention, query-conditioned retrieval, sparse top-k routing,
straight-through estimators, iterative refinement, and value functions all
have substantial prior art. QVESR is not presented as a novel ingredient. It
is the fastest discriminating test of the credit-assignment failure exposed by
CGSGR. A broader claim would require a capability win and a more specific
separator from those families.

## Matched pilot

Three arms instantiate and execute the same modules and parameter count:

- `UTILITY`: hard top-k learned query-conditioned evidence value;
- `FIXED`: cyclic evidence coverage;
- `RESIDUAL`: largest current consequence residual, reproducing CGSGR's rule.

All use seed 29, 1,000 updates, 256 examples/update, four recurrent rounds,
two probes/round, two revised state slots/round, and the frozen depth-5/7
noncommuting, repeated-binding, and induction cohorts. The pilot advances only
if UTILITY:

1. beats FIXED by at least five absolute macro points;
2. improves the mean of both depths in every family;
3. loses at least three macro points when selected evidence outcomes are
   shuffled;
4. loses at least three macro points when the selector alone receives a
   row-shuffled query; and
5. retains exact hard evidence and sparse-write budgets.

A miss closes QVESR and the nearby sparse-revision selector family. It does
not authorize another duration, width, seed, temperature, loss, or top-k
variant.
