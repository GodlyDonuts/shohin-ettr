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
