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
