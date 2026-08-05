# Counterexample-Guided Sparse Global Revision

Status: bounded mechanics pilot; no reasoning claim yet.

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
