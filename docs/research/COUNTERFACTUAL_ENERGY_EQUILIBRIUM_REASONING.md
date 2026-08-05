# Counterfactual Energy Equilibrium Reasoning

Status: one bounded transition-mechanism pilot; no reasoning or novelty claim.

## Hypothesis

PCSD, FCPT, CGSGR, and QVESR all manipulate evidence or state through learned
additive proposals. Their modules are causally active, yet better local
consistency does not become better composition. Counterfactual Energy
Equilibrium Reasoning (CEER) changes the transition itself.

A complete latent state predicts every prompt-owned source consequence. Their
masked negative log likelihood defines a scalar energy:

```text
E(S; X) = mean_j -log p(y_j | S, probe_j)
g_t = dE / dS_t
S_(t+1) = S_t - D_theta(S_t, g_t) * normalized(g_t)
```

`D_theta` is a positive diagonal preconditioner, so it can change step size
and geometry but cannot reverse the local energy gradient. The late query is
absent from equilibrium inference. Reasoning is therefore inner optimization
of one model-owned state against source evidence, not a sequence of arbitrary
recurrent proposals.

## Prior-art boundary

Energy-based models, predictive coding, active inference, learned optimizers,
deep equilibrium networks, Hopfield networks, recurrent refinement, and
gradient-based latent inference are established. CEER does not claim those
ingredients as new. Its bounded question is whether differentiating a learned
prompt-owned consequence energy supplies a materially better compositional
transition than an ordinary recurrent updater under exact matched execution.

## Matched control

Every arm instantiates and executes both candidate transitions in every round:

- `ENERGY`: positive-preconditioned descent of consequence energy;
- `RECURRENT`: dense cross-attention to all evidence followed by an ordinary
  gated recurrent update.

The selected state is the only arm-dependent line. Parameter count, module
execution, recurrent rounds, examples, data, update count, and frozen cohorts
are identical. Both arms receive all evidence every round, eliminating sparse
selection as a confound.

The seed-37 pilot uses 1,000 updates, 256 examples/update, four inference
rounds, training depths 2--4, and frozen depth-5/7 noncommuting, binding, and
induction cohorts. ENERGY advances only if it:

1. beats RECURRENT by at least five absolute macro points;
2. improves the mean of both depths in every family;
3. reduces its own evidence energy across inference;
4. loses at least three macro points when evidence outcomes are shuffled; and
5. loses at least three macro points when its energy gradient is zeroed.

A miss closes this exact energy-transition mechanism. It does not authorize a
step-size, duration, width, seed, preconditioner, energy, or recurrence sweep.

## Result

The bounded seed-37 pilot is complete and CEER is closed.

| Family | Depth | ENERGY | RECURRENT | Delta |
|---|---:|---:|---:|---:|
| Noncommuting | 5 | 27.051% | 24.023% | +3.027 |
| Noncommuting | 7 | 50.684% | 52.051% | -1.367 |
| Binding | 5 | 18.945% | 20.898% | -1.953 |
| Binding | 7 | 15.039% | 15.332% | -0.293 |
| Induction | 5 | 9.863% | 9.961% | -0.098 |
| Induction | 7 | 9.766% | 8.105% | +1.660 |
| **Macro** | | **21.891%** | **21.729%** | **+0.163** |

Both arms have exactly 112,726 parameters, receive 256,000 training examples,
execute both transition modules at every round, and differ only at the selected
state transition. ENERGY reduces prompt-evidence energy by a mean 0.720 nats.
Its mechanism is causally active: shuffling evidence outcomes reduces macro
accuracy to 10.449% and zeroing the energy gradient reduces it to 10.661%.
Nevertheless, the final capability gain is only 0.163 points, it is negative
on three of six cohorts, and polynomial induction remains at chance. CEER
therefore misses both the +5-point and every-family gates.

Report SHA-256 values are:

- ENERGY: `9eaf0bed993d3a662b971131a8e1505ead902b8a274faa7f882863f2696d4868`
- RECURRENT: `e96a49c86badef46dba930faa4ed8541437249f9910e7cd5a1b18605b1ef5f76`

## Read-only interface diagnostic

CEER learns evidence predictions with `ConsequenceHead` but answers the late
query with a separate attention/readout head. A checkpoint-only diagnostic
applied the learned consequence head directly to the held-out query without
changing weights. This shared consequence readout reaches only 4.801% macro
for ENERGY and 4.867% for RECURRENT, versus 21.891% and 21.729% through the
separate readers. It scores 0% on both noncommuting cohorts and remains near
chance on induction.

Diagnostic SHA-256:
`f03ae392464a406a09052459b8124b509c2f229191c4a5d20349dc27a8639c2b`.

This rules out a post-hoc shared-head swap. The learned state/head pair can fit
observed consequences but does not encode one law that extends to an unseen
probe. The next bounded mechanism must make a single probe-conditioned law
responsible for both source consequences and the final query during training.
It must not add another selector, recurrence sweep, or unconstrained opaque
state reader.
