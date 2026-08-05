# Query-Valued Evidence Sparse Revision

Status: closed negative after one bounded credit-assignment pilot.

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

## Result

The first dispatch (`739242--739244`) failed before training because the
immutable runtime omitted one imported support module. It produced no reports
or checkpoints. The corrected fail-fast runtime has SHA256SUMS SHA-256
`7c2b11bac068ce8cc47273ecf9719c00d4c10a1f637d5ff70c2ff0a553c9ef3d`.

Valid jobs `739246--739248` complete with identical 120,983-parameter models:

| Family | Depth | Utility | Fixed | Residual |
|---|---:|---:|---:|---:|
| Noncommuting | 5 | 25.488% | 26.660% | 22.754% |
| Noncommuting | 7 | 29.590% | 29.492% | 32.910% |
| Binding | 5 | 20.605% | 20.117% | 18.066% |
| Binding | 7 | 14.355% | 15.137% | 13.477% |
| Induction | 5 | 9.570% | 9.961% | 9.961% |
| Induction | 7 | 9.375% | 9.277% | 8.887% |
| **Macro** | | **18.164%** | **18.441%** | **17.676%** |

UTILITY loses to FIXED by `0.277` points and fails the all-family requirement.
Shuffling selected outcomes lowers UTILITY to `10.726%`, confirming that the
revision still depends on evidence. Shuffling only the selector query lowers
it by merely `0.423` points to `17.741%`; the learned policy does not acquire
material query-specific value assignment. Contradiction reduction is
`0.608 / 0.288 / 0.779` for utility/fixed/residual, again showing that better
local correction does not imply better final composition.

QVESR and nearby sparse evidence-selector variants are closed. The next
architecture must change how structured state transitions and composes, not
retune evidence routing into this revision operator.

Report SHA-256 values:

- utility: `d50da2a6f7bdee8fb6ae951faeb2e1a1f18672ee0a35293fd04963d6bdc61f82`
- fixed: `7b895de5751b3c01486ee734ecd2d86520617cef1eedf3a9a0ffde023630c335`
- residual: `5cd99aefc42c1d4fdfc37ad6d49bb3d1121e6f4175b33bf8a1cb33dfc44ec7d3`
