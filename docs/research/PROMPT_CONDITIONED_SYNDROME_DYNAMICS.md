# Prompt-Conditioned Syndrome Dynamics

Status: architecture hypothesis frozen for a minimal matched falsifier.

## 1. Capability thesis

Shohin's recurring failure is not the absence of local operations. Across DRS,
typed-controller, sticky-schedule, integrated-workspace, and product-training
experiments, local binding or one-step competence often appears before exact
multi-step composition. The failure signature is accumulated state drift:
individually plausible edits combine into a globally inconsistent latent state.

**Prompt-Conditioned Syndrome Dynamics (PCSD)** treats the reasoning state as a
problem-specific error-correcting code. The source compiles a sticky set of
latent parity checks once. Every recurrent proposal is allowed to change the
state only after a differentiable syndrome projection restores those checks.
The checks are model-owned, receive no target/query/answer, and remain fixed for
the full trajectory.

The intended inductive bias is:

> Reasoning steps may change facts, but valid steps preserve task-specific
> invariants. Enforcing those invariants at every latent commit should prevent
> locally plausible operations from drifting into an incoherent composition.

This is a falsifiable mechanism claim, not a claim that error correction,
recurrence, latent reasoning, or neural code decoding is new. A 2025 recursive
latent-reasoning system already combines recurrence, discrete anchoring,
intermediate supervision, and random-corruption self-correction. PCSD differs
in the proposed object and operation: it compiles a source-conditioned linear
constraint geometry and performs an explicit minimum-norm projection after
every sparse transaction. A formal novelty claim remains contingent on a
broader literature review and a positive matched result.

## 2. Mechanism

Let the workspace at step `t` be `Z_t in R^(S x D)`. A prompt encoder produces
source features `X`. A parity compiler emits `C` normalized check factors:

```text
A(X) in R^(C x S),  B in R^(C x D)
P_c = A_c outer B_c
```

The sticky reference syndrome is:

```text
r_c = <P_c, Z_0>
```

A tied recurrent proposer emits a raw update `U_t` and sparse commit gates
`g_t`. The uncorrected state is:

```text
Z'_t = Z_t + g_t * U_t
e_t  = P Z'_t - r
```

With `G = P P^T + epsilon I`, PCSD applies the minimum-norm correction:

```text
Z_(t+1) = Z'_t - P^T G^-1 e_t
```

Therefore `P Z_(t+1) approximately r` at every step. The solve is over the
small `C x C` Gram matrix, is differentiable, and uses no host reasoning. The
answer head reads only the final corrected state after adaptive halt.

The architecture logs pre/post syndrome norm, correction energy, gate sparsity,
Gram conditioning, and halt depth. A zero-syndrome solution with no task gain is
not success.

## 3. What is held constant

All arms use the same:

- tokenizer/backbone or synthetic source encoder;
- workspace slots, width, recurrent steps, late readout, and answer loss;
- train/development/confirmation examples and order;
- optimizer, charged tokens, update count, and random seeds;
- source access and adaptive halt contract;
- parameter and measured training-FLOP budget within 1%;
- model-owned inference with no external host, tool, verifier, or search.

Arms:

1. `PCSD`: prompt-conditioned sticky checks plus syndrome projection.
2. `DENSE`: parameter-matched untied dense correction after each proposal.
3. `RECURRENT`: strongest tied recurrent workspace without correction, padded
   with live capacity if needed to match parameters/FLOPs.
4. `PCSD-SHUFFLED`: check geometry from a different prompt in the minibatch.
5. `PCSD-ZERO`: correction disabled at evaluation with all weights unchanged.

## 4. Fast discriminating benchmark

The first gate is a generated Conservation-Shift Suite, not a public benchmark.
It contains several renderer-randomized register systems whose commands are
sparse but whose valid trajectories preserve source-defined affine invariants.
The model receives only the rendered world, command stream, and late query.

Training uses command depths 2--8. Development uses new values/renderers at
depths 2--8. Confirmation is generated and sealed before training and tests:

- depth 16 and 32;
- unseen entity permutations and group sizes;
- unseen compositions of individually trained operations;
- a one-step latent perturbation at a hidden random trajectory position.

Metrics are exact terminal state, exact queried answer, invariant violation,
corruption recovery, and accuracy by depth. Architecture selection never reads
the confirmation labels.

## 5. Pass and kill rules

PCSD passes the minimal gate only if all three fixed seeds satisfy:

1. depth-16 exact answer is at least 70% and depth-32 at least 50%;
2. mean confirmation exact answer is at least 15 absolute points above both
   `DENSE` and `RECURRENT` under matched budgets;
3. latent-perturbation recovery is at least 25 points above both controls;
4. `PCSD-ZERO` and `PCSD-SHUFFLED` each lose at least 10 points;
5. post-projection syndrome is at least 100x smaller than pre-projection
   syndrome without a degenerate zero-update policy;
6. parameter mismatch and measured training-FLOP mismatch are each <=1%.

Kill PCSD after this one cohort if it misses the matched-control margin, if the
controls tie it, if the ablations do not remove the gain, or if improvement is
restricted to invariant violation without exact-answer transfer. A clean pass
advances to language-backed arithmetic/logic and then a scratch small-model
integration; it does not by itself establish general reasoning.

## 6. Immediate implementation sequence

1. Implement the standalone sticky check compiler and minimum-norm projector.
2. Unit-test exact syndrome reduction, gradients, source-only ownership,
   shuffled checks, and deterministic replay.
3. Implement the Conservation-Shift generator and seal confirmation hashes.
4. Implement the shared trainer with exact parameter/FLOP receipts.
5. Run one smoke seed, then the fixed three-seed matched cohort only if finite.

The frozen Qwen-hosted product route remains a practical control. It is not an
arm in this architectural claim and supplies no inference-time component.
