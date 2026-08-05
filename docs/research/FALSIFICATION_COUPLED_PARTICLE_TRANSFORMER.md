# Falsification-Coupled Particle Transformer

Status: mechanics implementation; no reasoning claim yet.

## Thesis

Shohin's repeated failure is not simply too little recurrent depth. Several
runs learn useful local fields but settle into incompatible complete programs;
softly averaging those fields destroys strict causality. FCPT therefore treats
a complete candidate world/program as the indivisible inference unit.

FCPT maintains `K` exchangeable particles

```text
P_i^t = (S_i^t, log w_i^t, lineage_i^t)
```

where `S_i` is a complete structured state. One shared recurrent proposer
creates `B` complete branch candidates per particle:

```text
S_tilde_(i,b) = T_theta(S_i, source, challenge, branch_b)
```

Each candidate predicts outcomes for source-owned evidence probes. The
contradiction bus admits only `R` probes with high behavioral disagreement:

```text
q* = TopR_q Var_(i,b)[p_theta(y_q | S_tilde_(i,b))]
```

Observed source evidence updates candidate credibility with a proper log
score. The next population is selected as complete states:

```text
log w_tilde_(i,b) = log w_i + mean_(q in q*) log p(y_q | S_tilde_(i,b))
P^(t+1) = WholeTopK({S_tilde_(i,b), log w_tilde_(i,b), lineage_(i,b)})
```

No opcode, pointer, register, relation, or latent field can be copied from a
different lineage. The late reader consumes one winning complete state. Query
and answer targets are absent from proposal, contradiction, and selection
APIs.

## Honest novelty boundary

Particles, recurrent depth, latent reasoning, global workspaces,
differentiable particle filters, counterexample-guided synthesis, and hard
selection all have prior art. FCPT's candidate contribution is their specific
composition as model-owned falsification-coupled sequential Monte Carlo over
structured latent programs, with persistent whole-program identity and no
fieldwise aggregation.

This is only a credible contribution if the learned contradiction channel is
causal. Merely sampling several states or choosing the best one is a control,
not FCPT.

## Minimal implementation

`train/falsification_coupled_particles.py` currently provides:

- exchangeable particle-axis operations with shared branch proposals;
- complete-state gathering and explicit lineage tracking;
- behavioral outcome prediction over source evidence;
- a fixed-bandwidth disagreement-selected contradiction bus;
- evidence log-score updates and whole-state top-K selection;
- independent-particle, soft-aggregation, and selection-without-falsification
  controls from the same components;
- a late query reader over one state; and
- behavioral-equivalence certification over predicted consequences.

The first mechanics suite requires finite gradients, selection invariance
under particle permutation, preserved nonnegative lineage, no query in the
deliberation signature, and a soft-control marker proving where identity is
intentionally destroyed.

## Three-family gate

All families expose evidence in the source; no hidden answer, tool, teacher,
or external verifier is available during inference.

1. **Ambiguous noncommuting programs.** Several operation orders explain early
   local observations; later input/output evidence distinguishes the complete
   order. OOD changes depth, operation combinations, symbols, and renderer.
2. **Repeated-binding graph programs.** Candidate entity bindings agree on
   local attributes but differ on multi-hop consequences. OOD changes graph
   size, repeated-name frequency, relation composition, and renderer.
3. **Counterexample-guided function induction.** Several functions fit an
   initial example prefix. Additional source examples falsify alternatives;
   the late query asks for a new value. OOD changes function families,
   coefficient ranges, and number of distractor hypotheses.

Training uses depths 2--4 and cardinalities 3--6. Development uses depths
5--7 and cardinalities 7--9. Confirmation uses a separately frozen generator,
renderer, depth range, and seed set that training jobs cannot load.

## Matched controls

- ordinary Transformer;
- one tied recurrent stream;
- `K` independent recurrent particles;
- particles with soft latent aggregation;
- whole-particle selection with fixed probes and no disagreement bus;
- full FCPT;
- FCPT with shuffled contradiction messages;
- FCPT with lineage swapped before the late reader.

The single-stream recurrent control receives the same total proposal calls as
all FCPT particles and branches combined. Results report both trainable
parameters and measured forward/backward FLOPs; parameter and total-FLOP
mismatches must each remain within 1% for the primary comparison.

## Pass and kill rule

FCPT advances only if:

1. exact OOD joint accuracy is at least 10 absolute points above the best
   matched control across all three families;
2. every family improves and at least four of five fixed seeds improve;
3. removing the learned falsifier costs at least five points;
4. soft aggregation is materially worse than whole-particle selection;
5. shuffled challenges and lineage swaps cause the predicted degradation;
6. randomized counterfactual labels remove the gain;
7. useful particle count and behavioral diversity stay above one; and
8. the unopened confirmation passes without threshold changes.

Kill the lane if a compute-matched deep recurrent stream ties it, particles
collapse behaviorally, the scorer learns labels rather than evidence, or the
effect does not transfer across all three generated families.

## Resource envelope

The mechanics implementation is CPU-testable. The first GPU pilot uses two
single H100 jobs: FCPT and the strongest recurrent control, one seed, 1,000
updates, and development cohorts only. Budget: at most two H100-hours.

Only a >=5-point pilot trend with noncollapsed particles authorizes the full
development matrix. Six principal arms by five seeds require 30 independent
single-H100 jobs. The initial ceiling is 30 H100-hours; jobs stop early for
chance-flat or collapsed behavior. Ablations and confirmation are launched
only for a development winner. No scratch language pretraining begins from
this gate alone.

## Pilot v1 result and objective correction

Seed-17 jobs `739194` (FCPT) and `739195` (fixed-probe whole selection) used
identical 88,727-parameter models, 1,000 updates, 256 examples per update, and
the six frozen depth-5/7 development cohorts. FCPT reaches 12.174% macro exact
versus 11.475%, only +0.699 points. It loses slightly on depth-5 binding and
ties depth-7 induction. This misses the fixed +5 pilot trend and does not
authorize the full matrix.

The failure exposes an objective contradiction rather than a duration issue.
V1 applies gold behavior cross-entropy independently to every candidate. That
loss explicitly trains every particle to make the same predictions, while a
small diversity term tries to separate them. Final hard behavior uniqueness is
only 1.2--1.5 among eight branch candidates and mutual-information diversity
is about 0.0005--0.0007. The falsifier cannot allocate useful contradiction
bandwidth when its candidate set has already collapsed.

V1 is frozen. One corrected pilot replaces all-candidate behavior CE with a
multiple-hypothesis coverage objective: for each episode and round, only the
lowest complete-candidate evidence NLL is charged. A stronger behavioral
mutual-information term keeps the uncharged alternatives distinct. Data,
architecture, updates, seed, evaluation, and FCPT-versus-selection comparison
remain fixed. This is the only authorized successor. It must still clear +5
points and noncollapse; otherwise FCPT closes before the full matrix.

V1 report SHA-256 values:

- FCPT: `09fc81a4aa0c36153b949e735ff7458d28901e9ac88b3da9ab4296bb1d7226d0`
- selection: `3c03af58878f75a57d1907c94cc8e5d4b2dfd82bf34b88d5ce40f72bcad85e63`

## Pilot v2 result: FCPT closed

The coverage objective succeeds at its intended intermediate effect. FCPT
candidate uniqueness rises to 3.06--4.28 of eight on development, and the
selection control reaches 2.91--4.39. Behavioral mutual information rises by
roughly three orders of magnitude to 0.8--1.2 during training. Posterior
collapse is no longer the primary failure.

Capability still does not clear the gate. FCPT macro exact is 12.093% versus
11.800% for fixed-probe whole selection, only +0.293 points. FCPT loses on
depth-5 binding and depth-7 induction and remains close to chance on induction.
The learned disagreement bus therefore does not produce a material advantage
even when distinct hypotheses exist.

No additional FCPT widths, durations, seeds, losses, or full controls are
authorized. The lane closes before confirmation. The next architecture target
is a single coherent but globally revisable state whose own prediction errors
select sparse counterexample-conditioned updates, avoiding both particle
collapse and premature whole-hypothesis selection.

V2 report SHA-256 values:

- FCPT: `18cc9e5175979b954ff1c4311404638fd840216846bf4cf7ac545cab80afa536`
- selection: `2368c2055e365c239af6308040cb7d18dc55becd2d2f6d22958c73b2c3df84ac`
