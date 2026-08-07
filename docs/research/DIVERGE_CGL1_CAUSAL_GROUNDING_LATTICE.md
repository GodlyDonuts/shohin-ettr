# DIVERGE-CGL1: Causal Grounding Lattice

Status: data mechanics frozen after the QTE1 development ceiling and before
any CGL1 neural training. QTE1 confirmation and composition remain independent.

## Hypothesis

GTI1 demonstrates that direct role labels are cheaply fit through renderer
lookup. CGL1 removes those labels from the candidate-visible corpus. It
represents both complete `READ` transactions, executes each against a sealed
two-candidate state, and supervises only the observed terminal answer:

\[
  L_{outcome} = -\log \sum_{j: execute(T_j)=y^*} p(T_j\mid source,state).
\]

Each semantic source appears under three state interventions: two distinct
value assignments that swap the target outcome, and one equal-outcome state
where both transactions produce the same answer. The two clause orders for
each meaning are also present. A semantic transaction must therefore stay
coherent across six records even when the outcome alone is underdetermined.

This is not claimed as novel latent-variable learning. Its purpose is to test
whether downstream consequences plus intervention consistency can train a
small model-owned semantic owner after direct role fitting failed. QTE1 is the
fixed 0.8B capability ceiling, not a runtime teacher in this gate.

## Frozen data contract

The source is the immutable 100,000-row RRG1 QUERY corpus at SHA-256
`2d325c860e707307886f782350e7ec35ae8c23ae275260b0a937bbb738078c1c`.
The deterministic builder emits 300,000 public records and 300,000 separate
outcome-supervisor records: 200,000 distinct-outcome and 100,000 equal-outcome
cases across 50,000 complete semantic pairs.

Public records contain source text, source-owned symbols, anonymous candidate
values, state-orbit identity, and commitments. They contain no target,
distractor, symbol-role, role-order, or gold-transaction field. Supervisors
contain only the committed public identity and terminal answer. Exhaustive
generation verifies every six-record orbit, value swap, equal-outcome case,
clause-order answer invariant, identity, and forbidden-field audit.

The first local exhaustive build produced provisional hashes:

- public: `bc438f793a3ced67a3b5493d70c14cbc39db4c20f3fe0fb50579af6b5f1daea9`;
- supervisor: `affa2cc36412f07f2816a00bbe2abfb06ee93be3b602c79b79b248f4ccf2552d`;
- report: `9823ba87ba37e710800701760e12486c701a281bf1d630ce019bbc7aee8d49ee`.

Stokes must reproduce those hashes before neural admission.

## Neural admission boundary

No neural CGL1 run starts until QTE1 closes and a trainer/control contract is
frozen. At minimum, treatment must be compared against the closed GTI1 direct
role result and a matched flipped-outcome control. A small-model pass requires
at least `765/768`, every mode `254/256`, every renderer `127/128`, mapped
mention-swap equivariance `765/768`, and flipped control at most `430/768`.
Equal-outcome transactions must agree with their paired distinct-outcome
orbits rather than collapse to arbitrary state values. A miss closes the exact
outcome mechanism without seed, width, duration, renderer, or threshold
variants.
