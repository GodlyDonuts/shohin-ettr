# Counterexample-Guided Sparse Global Revision

Status: closed negative after the bounded mechanics pilot.

## Thesis

PCSD preserved the wrong quantity and FCPT preserved alternatives without
using them effectively. CGSGR instead maintains one coherent mutable program
state. At every round, that state predicts the consequences of all source
evidence. Its own largest prediction residuals become model-owned
counterexamples, and a sparse write mask revises only the implicated state
slots.

For state `S_t`, source probes `q_j`, and observed source outcomes `y_j`:

```text
p_j = ConsequenceHead(S_t, q_j)
r_j = -log p_j[y_j]
J_t = TopR_j r_j
c_t = Encode({q_j, y_j, r_j : j in J_t})
m_t = SparseSlotRoute(S_t, c_t)
S_(t+1) = S_t + m_t * Revision(S_t, c_t)
```

The query is absent until all revisions finish. The selected evidence is part
of the prompt, not a hidden label or external verifier. The state remains one
complete lineage, but early latent commitments can be changed in place.

## Prior-art boundary

Iterative refinement, masked diffusion, recurrent depth, sparse writes,
counterexample-guided synthesis, and verifier feedback are established
families. The candidate separator is an internal closed loop where a language-
conditioned state predicts prompt-owned consequences, ranks its own errors,
and uses those exact counterexamples to route sparse global revisions before a
late query. This is a hypothesis, not a novelty claim.

## Matched pilot

Both arms share every parameter, update, example, loss, recurrent round,
revision bandwidth, and write bandwidth:

- `GUIDED`: chooses the two source probes with largest current contradiction;
- `FIXED`: visits source probes on a deterministic cyclic schedule.

The pilot reuses the frozen depth-5/7 noncommuting, binding, and induction
cohorts. Training is depth 2--4 for 1,000 updates. The first gate requires:

1. at least +5 absolute macro exact over FIXED;
2. nonnegative gain on every family;
3. larger contradiction reduction than FIXED;
4. shuffled counterexample outcomes lose at least three points; and
5. nonzero sparse revisions with exactly the fixed slot budget.

A miss closes this exact mechanism. A pass authorizes independent/dense/deep-
recurrent controls and a newly sealed confirmation generator. It does not
authorize language pretraining by itself.

## Implementation

- `train/counterexample_guided_revision.py`: source-only initialization,
  consequence prediction, residual selection, sparse slot routing, shared
  revision, late query readout, and shuffled-outcome ablation.
- `train/cgsgr_plural_reasoning.py`: the matched three-family trainer and
  hash-bound evaluator.
- `train/jobs/cgsgr_plural_reasoning.sbatch`: isolated one-H100 arm launcher.

The mechanics tests require exact sparse-write cardinality, equal treatment
and control parameter counts, finite gradients, query-late ownership, and the
predicted state change under shuffled counterexamples.

## Result

Jobs `739226` (`GUIDED`) and `739227` (`FIXED`) used seed 23, 1,000
updates, 256 examples/update, identical 108,438-parameter models, and the
same frozen depth-5/7 cohorts. Exact answer accuracy was:

| Family | Depth | Guided | Fixed | Delta |
|---|---:|---:|---:|---:|
| Noncommuting | 5 | 25.879% | 25.977% | -0.098 |
| Noncommuting | 7 | 35.840% | 33.789% | +2.051 |
| Binding | 5 | 17.578% | 18.750% | -1.172 |
| Binding | 7 | 12.109% | 14.453% | -2.344 |
| Induction | 5 | 9.961% | 10.059% | -0.098 |
| Induction | 7 | 8.496% | 9.766% | -1.270 |
| **Macro** | | **18.311%** | **18.799%** | **-0.488** |

The mechanism is active but misaligned. Guided revision reduces mean source
contradiction by `1.209`, versus `0.663` for fixed revision. Shuffling the
selected counterexample outcomes drops guided macro accuracy from `18.311%`
to `9.782%` (`-8.529` points), proving that the state update materially uses
the selected evidence. Nevertheless, choosing the largest current residual
hurts final-answer exactness and loses on four of six cohorts.

The likely failure is credit assignment: the most surprising or locally
incorrect consequence is not necessarily the evidence with greatest value
for the requested answer. Raw-residual CGSGR therefore closes. There is no
duration, width, seed, or loss extension. A successor must learn final-answer
value of evidence rather than reuse contradiction magnitude as a proxy.

Report SHA-256 values:

- guided: `00481eaf101b48d5e56f2254a7a7cf41539a5f1ae39ed3db48b846da998c2853`
- fixed: `28fbc6cd75a1df717893541097035196ed62e7afbf9525b24af45d33cc8b9cbe`
