# R12 EFC Counterfactual Machine-Repair Lattice

## Status

**REJECTED BEFORE FIT.** CMRL is retained as a negative architecture-mechanics
record only. It is not part of JASEC's parameter receipt, is not admitted for a
neural run, and is not evidence of reasoning. It does not authorize
continuation pretraining. The protected step-300k Shohin checkpoint remains
immutable.

Independent hostile review found that the prototype's machine-shaped
`transition_evidence` and `observer_evidence` were treated as the exact target
machine. A candidate feature exposed the target probability directly. The
leave-one-out control preserved the same target through the invertible map
`target[c] = 1 - (K-1) mean_others(c)`, while the observational twin collapsed
all candidate vectors to equality and therefore removed the categorical
choice channel. The fixed-cycle implementation did not disable row gates,
unsupported-cell custody was absent, and the final reported halt hazard did
not affect the mixture. Sixteen unit tests passed, but those tests established
tensor mechanics and equivariance only; they failed to test the scientific
boundary. This result closes the implementation below.

Any successor must use a compiler-owned, source-hash-bound constraint object
constructed from incomplete source observations. It must withhold the row
being repaired, prove at least one unresolved row remains unavailable, and
score interventions only against sealed observational trajectories. It may
not accept a complete machine-shaped evidence tensor. New matched controls
must preserve the candidate choice channel and the information multiset while
breaking only intervention-to-consequence alignment.

## Failure being targeted

The current Joint Assignment-Semantics Equilibrium Compiler (JASEC) improves
binding and machine semantics through four tied first-order corrections. That
architecture must infer both:

1. which finite machine edit is available; and
2. what downstream behavior the edit would cause.

The consumed ACSO result demonstrated why this matters: a locally plausible
cycle-zero causal gradient pointed toward the wrong destination on every one
of 672 deep faults. More width does not change that credit-assignment
geometry.

CMRL changes the computation. It treats source compilation as differentiable
model-predictive repair over a finite anonymous program lattice. Every legal
local categorical intervention is evaluated through fixed future-behavior
probes before the tied neural controller chooses a repair.

## Candidate computation

Let `A` be the soft physical-key assignment, `M=(T,O)` the supported soft
machine, and `X` anonymous source evidence. For each supported machine row
`i` and legal category `c`, form the finite intervention

`M[i <- c] = M + gamma_i * (one_hot(c) - M_i)`.

The architecture constructs a counterfactual residual

`Delta[i,c] = R(A, M[i <- c], X) - R(A, M, X)`.

`R` may use only candidate-visible anonymous quantities:

- assignment feasibility and confidence;
- witness transition and observation agreement;
- physical-key nerve compatibility;
- repeated-action and ordered noncommutative signatures;
- fixed action words through depth three;
- observer separation;
- reachability and collision syndromes; and
- base/derivative future-behavior agreement.

The public forward must not accept externally computed candidate scores,
oracle machines, hidden labels, late queries, or executor answers.
Counterfactuals are constructed internally by tensor operations over the
candidate machine and anonymous evidence.

One shared controller is reused across rows, categories, and up to eight
repair cycles:

`z_ic = E_delta(Delta_ic) + E_row(q_i) + W_h h_i`

`p_ic = softmax_c(f_theta(z_ic) / temperature)`

`M_i_next = retract((1-g_i) M_i + g_i sum_c p_ic one_hot(c))`.

The row memory `h_i` is updated from a symmetric pool over candidate
consequences. A model-owned halt hazard produces an adaptive-computation
mixture during training. Mechanics tests must not use hard data-dependent
Python early exit; deployment policy is a later protocol.

## Architectural departure

CMRL changes three assumptions usually left fixed:

1. **Forward-pass semantics:** the model evaluates finite interventions rather
   than only propagating the current activation.
2. **Computation depth:** a learned halt state allocates tied recurrent work
   by unresolved conflict rather than using one fixed transformer depth.
3. **Credit assignment:** finite downstream counterfactual consequences are
   explicit inputs to the repair controller, rather than being compressed
   into an infinitesimal local gradient.

This is closer to model-predictive control and program synthesis than to a
wider feed-forward decoder.

## Equivariance and custody

No state, action, observer, answer, physical-key, row, or candidate coordinate
embedding is permitted. Candidate-aligned features may distinguish the
proposed category from the symmetric pool of alternatives, but may not encode
its absolute index. The same controller and memory update are shared over all
coordinates.

For every state/action/observer/answer recoding `g`, the required contract is

`CMRL(gA, gM, gX) = g CMRL(A, M, X)`.

Opaque literals are absent from the trainable view. Their equality partition
may enter through JASEC's anonymous incidence bus. Raw key bytes remain
custody-only and are copied only after hard assignment.

CMRL exists only during attached source compilation. Before a late query:

- hard machine fields and copied hard keys are sealed;
- counterfactual lattices and row memories are destroyed;
- anonymous source tensors and frozen residuals are destroyed; and
- the detached query parser receives only the sealed wire and hard keys.

## Exact parameter budget

Default controller width is `D=704`; row-memory width is `H=384`.

| Component | Formula | Parameters |
|---|---:|---:|
| Counterfactual stem | `128D + D` | 90,816 |
| Row-context stem | `64D + D` | 45,760 |
| Memory initializer | `32H + H` | 12,672 |
| Two 4D residual blocks | `2(8D^2 + 7D)` | 7,939,712 |
| Shared GRU cell | `3HD + 3H^2 + 6H` | 1,255,680 |
| Memory projection | `HD` | 270,336 |
| Candidate readout | `2D + D + 1` | 2,113 |
| Row repair gate | `D + H + 1` | 1,089 |
| Halt head | `H + 1` | 385 |
| Four temperatures/scales | `4` | 4 |
| **CMRL total** |  | **9,618,567** |

The rejected prototype would have produced the following hypothetical receipt:

| Component | Parameters |
|---|---:|
| Frozen Shohin | 125,081,664 |
| Gauge-invariant JASEC compiler | 63,671,588 |
| Detached query parser | 748,033 |
| Rejected CMRL prototype | 9,618,567 |
| **Hypothetical complete system** | **199,119,852** |
| **Hypothetical headroom below 200M** | **880,148** |

These parameters are not admitted. Current JASEC remains 189,501,285 complete
parameters with 10,498,715 available below the strict 200,000,000 limit.

## Matched controls

1. **Observational twin:** rejected because candidate averaging made every
   candidate vector identical and structurally removed categorical choice.
2. **Candidate averaged:** rejected because leave-one-out averaging is an
   invertible encoding of a normalized candidate target.
3. **One-step myope:** remove every depth-two/depth-three consequence while
   retaining immediate transition/observer evidence.
4. **Commutative counterfactual:** identify probe words by action multiset,
   deleting order while retaining length and frequency.
5. **Unsigned repair:** replace signed counterfactual improvement by its
   magnitude.
6. **Fixed-cycle repair:** intended to disable adaptive halt and row gates,
   but the prototype disabled only halt selection and therefore failed.
7. **Equal-parameter widening:** spend the same budget on an observational
   controller without intervention consequences.

## Mechanics gates

Before any fit:

- exact parameter receipt below the global limit;
- no raw-source, raw-key, target-machine, label, late-query, or answer input in
  the public API;
- finite forward and nonzero finite backward;
- every supported categorical row remains normalized;
- unsupported cells remain exactly unavailable;
- state/action/observer/answer and physical-key recodings commute with every
  cycle, final probabilities, gradients, and hard machine;
- candidate averaging preserves dimensions, parameters, and compute while
  breaking candidate identity;
- one-step mode retains one-transition evidence and excludes ordered paths;
- source deletion and post-seal source poisoning cannot change execution; and
- a fixed nonlearned counterfactual oracle must recover every existing deep
  fault, or the candidate lattice is information-insufficient and CMRL is
  rejected before fitting.

## Closure decision

No fit, score, or reasoning claim may be produced from this implementation.
The useful surviving conjecture is narrower: finite intervention search may
help only when consequences are evaluated against incomplete, source-derived
behavioral constraints that cannot reconstruct the hidden machine. That
requires a new name, protocol, implementation, and hostile review.
